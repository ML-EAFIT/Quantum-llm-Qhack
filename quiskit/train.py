"""
train.py — Quantum Transformer trained on Shakespeare (Qiskit edition)
=======================================================================
Architecture
  - Classical token & positional embeddings (vocab → d_model)
  - Quantum Attention Block  : QFT-based Fourier mixer built with Qiskit
  - Classical Feed-Forward   : two linear layers with ReLU
  - Classical LM-head        : d_model → vocab, cross-entropy loss

Gradients
  Qiskit has no built-in autodiff, so the quantum layer is wrapped as an
  `autograd` primitive whose backward pass uses the parameter-shift rule
  (exact for RY / RZ gates).  The rest of the model is differentiated by
  autograd as before.

Requirements
  pip install qiskit autograd numpy        (qiskit >= 1.3 for QFTGate)

Usage
  python train.py
  → saves checkpoint.npz (weights) and vocab.json
"""

import os, json, time, urllib.request
import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.extend import primitive, defvjp

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFTGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# ──────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────
SEQ_LEN = 16  # context window (tokens)
D_MODEL = 8  # embedding dimension  (must be even)
N_QUBITS = D_MODEL // 2  # qubits for the quantum mixer
FF_DIM = 32  # feed-forward hidden size
BATCH = 32  # sequences per gradient step (lower this if training is slow)
EPOCHS = 300  # training epochs
LR = 0.02  # learning rate (Adam)
VOCAB_FILE = "vocab.json"
CKPT_FILE = "checkpoint.npz"
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
MAX_CHARS = 80_000  # subset for speed

# ──────────────────────────────────────────────────────────
# 1. Data
# ──────────────────────────────────────────────────────────


def fetch_shakespeare():
    path = "shakespeare.txt"
    if not os.path.exists(path):
        print("Downloading Shakespeare …")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()[:MAX_CHARS]


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def make_batches(tokens, seq_len, batch_size):
    """Yield (X, Y) arrays of shape (batch, seq_len)."""
    n = len(tokens) - seq_len
    arr = np.array(tokens, dtype=np.int32)
    while True:
        idxs = np.random.randint(0, n, batch_size)
        X = np.stack([arr[i : i + seq_len] for i in idxs])
        Y = np.stack([arr[i + 1 : i + seq_len + 1] for i in idxs])
        yield X, Y


# ──────────────────────────────────────────────────────────
# 2. Quantum Fourier Mixer (Qiskit)
# ──────────────────────────────────────────────────────────


def build_circuit():
    """
    Parameters θ[0:N]   → input angles (RY embedding of one token's half-vector)
                θ[N:2N] → learnable RZ angles

    Circuit:
      1. RY(x_i) on each qubit        (PennyLane AngleEmbedding, rotation="Y")
      2. RZ(w_i) on each qubit        (learnable phases)
      3. QFT on all qubits
      4. Measure <Z_i>
    """
    theta = ParameterVector("θ", 2 * N_QUBITS)
    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        qc.ry(theta[i], i)
    for i in range(N_QUBITS):
        qc.rz(theta[N_QUBITS + i], i)
    # PennyLane treats wire 0 as the most-significant qubit; Qiskit treats
    # qubit 0 as least-significant.  Reversing the qubit order reproduces
    # PennyLane's QFT exactly.
    qc.append(QFTGate(N_QUBITS), list(reversed(range(N_QUBITS))))
    return qc


def _z_observables():
    """<Z_i> for each qubit i (Qiskit Pauli labels are little-endian)."""
    obs = []
    for i in range(N_QUBITS):
        label = "I" * (N_QUBITS - 1 - i) + "Z" + "I" * i
        obs.append(SparsePauliOp(label))
    return obs


CIRCUIT = build_circuit()
OBSERVABLES = _z_observables()
ESTIMATOR = StatevectorEstimator()  # exact; swap for a Runtime estimator on hardware


def _run_batch(params_2d):
    """params_2d: (M, 2*N_QUBITS) → expectation values (M, N_QUBITS)."""
    pv = params_2d[:, None, :]  # (M, 1, P) broadcasts against observables (N_QUBITS,)
    result = ESTIMATOR.run([(CIRCUIT, OBSERVABLES, pv)]).result()
    return np.asarray(result[0].data.evs, dtype=np.float64)


def _build_params(x, w):
    return np.concatenate([x, np.broadcast_to(w, x.shape)], axis=1)


@primitive
def quantum_fourier_mixer(x, w):
    """
    x : (N, N_QUBITS) — one row per token (RY input angles)
    w : (N_QUBITS,)   — shared learnable RZ angles
    returns (N, N_QUBITS) of <Z> expectation values.
    """
    return _run_batch(_build_params(x, w))


# ---- backward pass: parameter-shift rule ------------------
_jac_cache = {}


def _jacobian(x, w):
    """d<Z_j>/dθ_p  with shape (N, N_QUBITS, 2*N_QUBITS), via parameter shift."""
    key = (x.tobytes(), w.tobytes())
    if key in _jac_cache:
        return _jac_cache[key]

    base = _build_params(x, w)
    N, P = base.shape
    shifted = np.empty((P, 2, N, P))
    shifted[:] = base
    for p in range(P):
        shifted[p, 0, :, p] += np.pi / 2
        shifted[p, 1, :, p] -= np.pi / 2

    evs = _run_batch(shifted.reshape(-1, P)).reshape(P, 2, N, N_QUBITS)
    jac = 0.5 * (evs[:, 0] - evs[:, 1])  # (P, N, N_QUBITS)
    jac = np.transpose(jac, (1, 2, 0))  # (N, N_QUBITS, P)

    _jac_cache.clear()
    _jac_cache[key] = jac
    return jac


defvjp(
    quantum_fourier_mixer,
    lambda ans, x, w: lambda g: np.einsum(
        "nj,njp->np", g, _jacobian(x, w)[:, :, :N_QUBITS]
    ),
    lambda ans, x, w: lambda g: np.einsum(
        "nj,njp->p", g, _jacobian(x, w)[:, :, N_QUBITS:]
    ),
)


def quantum_mix_sequence(seq_emb, rot_weights):
    """
    seq_emb    : (B, T, D_MODEL)
    rot_weights: (N_QUBITS,)

    The first half of each token embedding goes through the quantum mixer;
    the second half passes through unchanged.
    """
    B, T, _ = seq_emb.shape
    x_half = anp.reshape(seq_emb[:, :, :N_QUBITS], (B * T, N_QUBITS))
    r_half = seq_emb[:, :, N_QUBITS:]
    q_out = anp.reshape(quantum_fourier_mixer(x_half, rot_weights), (B, T, N_QUBITS))
    return anp.concatenate([q_out, r_half], axis=-1)


# ──────────────────────────────────────────────────────────
# 3. Model parameters
# ──────────────────────────────────────────────────────────


def init_params(vocab_size, seed=42):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.1, shape)

    return {
        "tok_emb": r(vocab_size, D_MODEL),
        "pos_emb": r(SEQ_LEN, D_MODEL),
        "q_rot": r(N_QUBITS),
        "ff_w1": r(D_MODEL, FF_DIM),
        "ff_b1": r(FF_DIM),
        "ff_w2": r(FF_DIM, D_MODEL),
        "ff_b2": r(D_MODEL),
        "ln1_g": np.ones(D_MODEL),
        "ln1_b": np.zeros(D_MODEL),
        "ln2_g": np.ones(D_MODEL),
        "ln2_b": np.zeros(D_MODEL),
        "lm_w": r(D_MODEL, vocab_size),
        "lm_b": r(vocab_size),
    }


# ──────────────────────────────────────────────────────────
# 4. Forward pass (batched: token_ids is (B, T))
# ──────────────────────────────────────────────────────────


def layer_norm(x, g, b, eps=1e-5):
    mean = anp.mean(x, axis=-1, keepdims=True)
    var = anp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / anp.sqrt(var + eps) + b


def relu(x):
    return anp.maximum(x, 0.0)


def forward(token_ids, params):
    """returns (B, T, vocab) log-probabilities"""
    T = token_ids.shape[1]
    x = params["tok_emb"][token_ids] + params["pos_emb"][:T]

    # Quantum attention block
    residual = x
    x_norm = layer_norm(x, params["ln1_g"], params["ln1_b"])
    x = residual + quantum_mix_sequence(x_norm, params["q_rot"])

    # Feed-forward block
    residual = x
    x_norm = layer_norm(x, params["ln2_g"], params["ln2_b"])
    h = relu(x_norm @ params["ff_w1"] + params["ff_b1"])
    x = residual + h @ params["ff_w2"] + params["ff_b2"]

    # LM head → log-softmax
    logits = x @ params["lm_w"] + params["lm_b"]
    m = anp.max(logits, axis=-1, keepdims=True)
    return logits - m - anp.log(anp.sum(anp.exp(logits - m), axis=-1, keepdims=True))


# ──────────────────────────────────────────────────────────
# 5. Loss
# ──────────────────────────────────────────────────────────


def cross_entropy_loss(params, X_batch, Y_batch):
    log_p = forward(X_batch, params)  # (B, T, V)
    B, T = Y_batch.shape
    correct = log_p[np.arange(B)[:, None], np.arange(T)[None, :], Y_batch]
    return -anp.mean(correct)


# ──────────────────────────────────────────────────────────
# 6. Adam optimiser
# ──────────────────────────────────────────────────────────


class Adam:
    def __init__(self, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        new_params = {}
        for k in params:
            g = grads[k]
            m = self.b1 * self.m.get(k, np.zeros_like(params[k])) + (1 - self.b1) * g
            v = (
                self.b2 * self.v.get(k, np.zeros_like(params[k]))
                + (1 - self.b2) * g**2
            )
            self.m[k], self.v[k] = m, v
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            new_params[k] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)
        return new_params


# ──────────────────────────────────────────────────────────
# 7. Training loop
# ──────────────────────────────────────────────────────────


def train():
    text = fetch_shakespeare()
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}  |  Data: {len(text)} chars")

    tokens = encode(text, stoi)
    params = init_params(vocab_size)
    optimiser = Adam(lr=LR)
    batcher = make_batches(tokens, SEQ_LEN, BATCH)
    grad_fn = grad(cross_entropy_loss, 0)

    print(
        f"\nTraining {EPOCHS} epochs  "
        f"(seq={SEQ_LEN}, d={D_MODEL}, qubits={N_QUBITS})\n"
    )
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Time/ep':>9}")
    print("-" * 30)

    for epoch in range(1, EPOCHS + 1):
        X_batch, Y_batch = next(batcher)
        t0 = time.time()
        loss = cross_entropy_loss(params, X_batch, Y_batch)
        grads = grad_fn(params, X_batch, Y_batch)
        params = optimiser.step(params, grads)
        elapsed = time.time() - t0

        if epoch % 20 == 0 or epoch == 1:
            print(f"{epoch:>6}  {float(loss):>8.4f}  {elapsed:>8.2f}s")

    save_dict = {k: np.array(v) for k, v in params.items()}
    save_dict["vocab_size"] = np.array(vocab_size)
    np.savez(CKPT_FILE, **save_dict)

    with open(VOCAB_FILE, "w") as f:
        json.dump({"stoi": stoi, "itos": {int(k): v for k, v in itos.items()}}, f)

    print(f"\n✓ Checkpoint saved → {CKPT_FILE}")
    print(f"✓ Vocabulary saved  → {VOCAB_FILE}")


if __name__ == "__main__":
    train()
