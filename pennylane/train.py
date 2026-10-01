"""
train.py — Quantum Transformer trained on Shakespeare
======================================================
Architecture
  - Classical token & positional embeddings (vocab → d_model)
  - Quantum Attention Block  : QFT-based mixing via PennyLane (replaces QKV)
  - Classical Feed-Forward   : two linear layers with ReLU
  - Classical LM-head        : d_model → vocab, cross-entropy loss

The quantum circuit implements a *Fourier mixer* (à la FNet) on
d_model//2  qubits using the PennyLane QFT + learnable RZ rotations,
producing a rich spectral mixing of token features without O(N²)
attention.

Usage
  python train.py
  → saves checkpoint.npz  (weights + vocab)
"""

import os, json, time, urllib.request
import pennylane as qml
import pennylane.numpy as pnp  # autograd-aware numpy
import numpy as np

# ──────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────
SEQ_LEN = 16  # context window (tokens)
D_MODEL = 8  # embedding dimension  (must be even)
N_QUBITS = D_MODEL // 2  # qubits for the quantum mixer
FF_DIM = 32  # feed-forward hidden size
BATCH = 32  # sequences per gradient step
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
# 2. Quantum Fourier Mixer (PennyLane)
# ──────────────────────────────────────────────────────────

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd")
def quantum_fourier_mixer(x_vec, rot_weights):
    """
    x_vec      : (N_QUBITS,) float  — encodes one token's feature slice
    rot_weights: (N_QUBITS,) float  — learnable per-qubit RZ angles

    Circuit:
      1. AngleEmbedding : maps x_vec → RY rotations on each qubit
      2. Learnable RZ   : parametrised single-qubit phase gates
      3. QFT            : full quantum Fourier transform
      4. Measure <Z>    : returns (N_QUBITS,) expectation values
    """
    qml.AngleEmbedding(x_vec, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS):
        qml.RZ(rot_weights[i], wires=i)
    qml.QFT(wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def quantum_mix_sequence(seq_emb, rot_weights):
    """
    seq_emb    : (seq_len, D_MODEL) — token embeddings
    rot_weights: (N_QUBITS,)        — shared learnable params

    For each token we run the quantum mixer on the *first half*
    of its embedding (N_QUBITS = D_MODEL//2) and return the
    measured ⟨Z⟩ values concatenated with the *second half*
    (residual connection in spectral space).
    """
    T, _ = seq_emb.shape
    rows = []
    for t in range(T):
        x_half = seq_emb[t, :N_QUBITS]  # first D/2 dims
        r_half = seq_emb[t, N_QUBITS:]  # second D/2 dims (passthrough)
        q_out = pnp.array(quantum_fourier_mixer(x_half, rot_weights))  # (N_QUBITS,)
        row = pnp.concatenate([q_out, r_half])  # (D_MODEL,)
        rows.append(row)
    return pnp.stack(rows)  # (T, D_MODEL)


# ──────────────────────────────────────────────────────────
# 3. Model parameters  (all in plain pennylane.numpy)
# ──────────────────────────────────────────────────────────


def init_params(vocab_size, seed=42):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return pnp.array(rng.normal(0, 0.1, shape), requires_grad=True)

    return {
        # Token embedding table: vocab_size × D_MODEL
        "tok_emb": r(vocab_size, D_MODEL),
        # Positional embedding: SEQ_LEN × D_MODEL
        "pos_emb": r(SEQ_LEN, D_MODEL),
        # Quantum mixer rotations: one RZ angle per qubit
        "q_rot": r(N_QUBITS),
        # Feed-forward layer 1 weights & bias
        "ff_w1": r(D_MODEL, FF_DIM),
        "ff_b1": r(FF_DIM),
        # Feed-forward layer 2 weights & bias
        "ff_w2": r(FF_DIM, D_MODEL),
        "ff_b2": r(D_MODEL),
        # Layer-norm (scale & shift) — two instances
        "ln1_g": pnp.ones(D_MODEL, requires_grad=True),
        "ln1_b": pnp.zeros(D_MODEL, requires_grad=True),
        "ln2_g": pnp.ones(D_MODEL, requires_grad=True),
        "ln2_b": pnp.zeros(D_MODEL, requires_grad=True),
        # LM head
        "lm_w": r(D_MODEL, vocab_size),
        "lm_b": r(vocab_size),
    }


# ──────────────────────────────────────────────────────────
# 4. Forward pass (single sequence, length T)
# ──────────────────────────────────────────────────────────


def layer_norm(x, g, b, eps=1e-5):
    mean = pnp.mean(x, axis=-1, keepdims=True)
    var = pnp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / pnp.sqrt(var + eps) + b


def relu(x):
    return pnp.where(x > 0, x, pnp.zeros_like(x))


def softmax(x):
    x = x - pnp.max(x, axis=-1, keepdims=True)
    e = pnp.exp(x)
    return e / pnp.sum(e, axis=-1, keepdims=True)


def forward(token_ids, params):
    """
    token_ids : (seq_len,) int  — input token indices
    returns   : (seq_len, vocab_size) log-probabilities
    """
    T = len(token_ids)
    # ── Embedding ──────────────────────────────────────
    x = params["tok_emb"][token_ids] + params["pos_emb"][:T]

    # ── Quantum Attention Block ─────────────────────────
    residual = x
    x_norm = layer_norm(x, params["ln1_g"], params["ln1_b"])
    x_mixed = quantum_mix_sequence(x_norm, params["q_rot"])
    x = residual + x_mixed  # residual

    # ── Feed-Forward Block ──────────────────────────────
    residual = x
    x_norm = layer_norm(x, params["ln2_g"], params["ln2_b"])
    h = relu(x_norm @ params["ff_w1"] + params["ff_b1"])
    x = residual + h @ params["ff_w2"] + params["ff_b2"]  # residual

    # ── LM Head → log-softmax ───────────────────────────
    logits = x @ params["lm_w"] + params["lm_b"]  # (T, vocab)
    log_p = (
        logits
        - pnp.log(
            pnp.sum(
                pnp.exp(logits - pnp.max(logits, axis=-1, keepdims=True)),
                axis=-1,
                keepdims=True,
            )
        )
        - pnp.max(logits, axis=-1, keepdims=True)
    )
    return log_p


# ──────────────────────────────────────────────────────────
# 5. Loss (cross-entropy, averaged over a mini-batch)
# ──────────────────────────────────────────────────────────


def cross_entropy_loss(params, X_batch, Y_batch):
    total = pnp.array(0.0)
    B = X_batch.shape[0]
    for b in range(B):
        log_p = forward(X_batch[b], params)  # (T, vocab)
        target = Y_batch[b]  # (T,)
        # gather log-probs of correct tokens
        correct = pnp.array([log_p[t, target[t]] for t in range(len(target))])
        total = total - pnp.mean(correct)
    return total / B


# ──────────────────────────────────────────────────────────
# 6. Adam optimiser (pure numpy / autograd)
# ──────────────────────────────────────────────────────────


class Adam:
    def __init__(self, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        new_params = {}
        for k in params:
            if grads.get(k) is None:
                new_params[k] = params[k]
                continue
            g = grads[k]
            m = self.m.get(k, pnp.zeros_like(params[k]))
            v = self.v.get(k, pnp.zeros_like(params[k]))
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g**2
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            self.m[k] = m
            self.v[k] = v
            new_params[k] = params[k] - self.lr * mh / (pnp.sqrt(vh) + self.eps)
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

    grad_fn = qml.grad(cross_entropy_loss, argnums=0)

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

    # ── Save checkpoint ────────────────────────────────
    save_dict = {k: np.array(v) for k, v in params.items()}
    save_dict["vocab_size"] = np.array(vocab_size)
    np.savez(CKPT_FILE, **save_dict)

    with open(VOCAB_FILE, "w") as f:
        json.dump({"stoi": stoi, "itos": {int(k): v for k, v in itos.items()}}, f)

    print(f"\n✓ Checkpoint saved → {CKPT_FILE}")
    print(f"✓ Vocabulary saved  → {VOCAB_FILE}")


if __name__ == "__main__":
    train()
