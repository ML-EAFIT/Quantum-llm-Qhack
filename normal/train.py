"""
train.py — Classical Transformer baselines on Shakespeare (no quantum)
======================================================================
Same architecture as the quantum version, with the quantum mixer swapped:

  --mixer attn     Causal single-head self-attention (a "normal" LLM block)
  --mixer fourier  Classical analog of the quantum circuit:
                   learnable phase -> DFT matrix -> tanh, on the same
                   D_MODEL//2 features, with the same N_QUBITS parameters.

Requirements
  pip install autograd numpy

Usage
  python train.py --mixer attn
  python train.py --mixer fourier
  -> saves checkpoint_<mixer>.npz, vocab.json, loss_<mixer>.json
"""

import os, json, time, argparse, urllib.request
import numpy as np
import autograd.numpy as anp
from autograd import grad

# ──────────────────────────────────────────────────────────
# Hyper-parameters (identical to the quantum train.py)
# ──────────────────────────────────────────────────────────
SEQ_LEN = 16
D_MODEL = 8
N_QUBITS = D_MODEL // 2  # kept for the Fourier analog (same feature count)
FF_DIM = 32
BATCH = 32
EPOCHS = 300
LR = 0.02
SEED = 0  # fixes batch order so every model sees the same data
VOCAB_FILE = "vocab.json"
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
MAX_CHARS = 80_000

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


def make_batches(arr, seq_len, batch_size, rng):
    n = len(arr) - seq_len - 1
    while True:
        idxs = rng.integers(0, n, batch_size)
        X = np.stack([arr[i : i + seq_len] for i in idxs])
        Y = np.stack([arr[i + 1 : i + seq_len + 1] for i in idxs])
        yield X, Y


# ──────────────────────────────────────────────────────────
# 2. Classical mixers
# ──────────────────────────────────────────────────────────

# Real/imag parts of the N-point DFT matrix (the classical "QFT")
_j, _k = np.meshgrid(np.arange(N_QUBITS), np.arange(N_QUBITS), indexing="ij")
DFT_C = np.cos(2 * np.pi * _j * _k / N_QUBITS) / np.sqrt(N_QUBITS)
DFT_S = np.sin(2 * np.pi * _j * _k / N_QUBITS) / np.sqrt(N_QUBITS)


def fourier_mix(x, phase):
    """
    Classical analog of  RY-embed -> RZ(w) -> QFT -> <Z>.
    First half of each token: phase-rotate, DFT, squash with tanh.
    Second half: passthrough (same as the quantum version).
    x: (B, T, D_MODEL), phase: (N_QUBITS,)
    """
    x_half = x[:, :, :N_QUBITS]
    r_half = x[:, :, N_QUBITS:]
    a = x_half * anp.cos(phase)
    b = x_half * anp.sin(phase)
    re = a @ DFT_C + b @ DFT_S  # Re[ DFT( x * e^{i*phase} ) ]
    return anp.concatenate([anp.tanh(re), r_half], axis=-1)


def softmax(x):
    x = x - anp.max(x, axis=-1, keepdims=True)
    e = anp.exp(x)
    return e / anp.sum(e, axis=-1, keepdims=True)


def attention_mix(x, p):
    """Causal single-head self-attention. x: (B, T, D_MODEL)."""
    T = x.shape[1]
    q, k, v = x @ p["wq"], x @ p["wk"], x @ p["wv"]
    scores = anp.einsum("btd,bsd->bts", q, k) / np.sqrt(D_MODEL)
    mask = np.tril(np.ones((T, T))) > 0
    scores = anp.where(mask, scores, -1e9)
    w = softmax(scores)
    return anp.einsum("bts,bsd->btd", w, v) @ p["wo"]


# ──────────────────────────────────────────────────────────
# 3. Parameters
# ──────────────────────────────────────────────────────────


def init_params(vocab_size, mixer, seed=42):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.1, shape)

    params = {
        "tok_emb": r(vocab_size, D_MODEL),
        "pos_emb": r(SEQ_LEN, D_MODEL),
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
    if mixer == "attn":
        for name in ("wq", "wk", "wv", "wo"):
            params[name] = r(D_MODEL, D_MODEL)
    elif mixer == "fourier":
        params["f_phase"] = r(N_QUBITS)
    else:
        raise ValueError(f"unknown mixer: {mixer}")
    return params


# ──────────────────────────────────────────────────────────
# 4. Forward pass (batched: token_ids is (B, T))
# ──────────────────────────────────────────────────────────


def layer_norm(x, g, b, eps=1e-5):
    mean = anp.mean(x, axis=-1, keepdims=True)
    var = anp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / anp.sqrt(var + eps) + b


def relu(x):
    return anp.maximum(x, 0.0)


def forward(token_ids, params, mixer):
    T = token_ids.shape[1]
    x = params["tok_emb"][token_ids] + params["pos_emb"][:T]

    # Mixing block (attention or Fourier)
    residual = x
    x_norm = layer_norm(x, params["ln1_g"], params["ln1_b"])
    if mixer == "attn":
        mixed = attention_mix(x_norm, params)
    else:
        mixed = fourier_mix(x_norm, params["f_phase"])
    x = residual + mixed

    # Feed-forward block
    residual = x
    x_norm = layer_norm(x, params["ln2_g"], params["ln2_b"])
    h = relu(x_norm @ params["ff_w1"] + params["ff_b1"])
    x = residual + h @ params["ff_w2"] + params["ff_b2"]

    logits = x @ params["lm_w"] + params["lm_b"]
    m = anp.max(logits, axis=-1, keepdims=True)
    return logits - m - anp.log(anp.sum(anp.exp(logits - m), axis=-1, keepdims=True))


# ──────────────────────────────────────────────────────────
# 5. Loss
# ──────────────────────────────────────────────────────────


def cross_entropy_loss(params, X_batch, Y_batch, mixer):
    log_p = forward(X_batch, params, mixer)
    B, T = Y_batch.shape
    correct = log_p[np.arange(B)[:, None], np.arange(T)[None, :], Y_batch]
    return -anp.mean(correct)


# ──────────────────────────────────────────────────────────
# 6. Adam
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


def train(mixer="attn", epochs=EPOCHS, batch=BATCH, verbose=True):
    text = fetch_shakespeare()
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    arr = np.array(encode(text, stoi), dtype=np.int32)

    split = int(0.9 * len(arr))  # 90% train / 10% validation
    train_arr, val_arr = arr[:split], arr[split:]

    params = init_params(vocab_size, mixer)
    n_params = int(sum(v.size for v in params.values()))
    optimiser = Adam(lr=LR)
    rng = np.random.default_rng(SEED)
    batcher = make_batches(train_arr, SEQ_LEN, batch, rng)

    loss_fn = lambda p, X, Y: cross_entropy_loss(p, X, Y, mixer)
    grad_fn = grad(loss_fn, 0)

    if verbose:
        print(f"\n[{mixer}] vocab={vocab_size}  params={n_params:,}")
        print(f"{'Epoch':>6}  {'Loss':>8}  {'Time/ep':>9}")
        print("-" * 30)

    history, t_start = [], time.time()
    for epoch in range(1, epochs + 1):
        X, Y = next(batcher)
        t0 = time.time()
        loss = loss_fn(params, X, Y)
        grads = grad_fn(params, X, Y)
        params = optimiser.step(params, grads)
        history.append(float(loss))
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"{epoch:>6}  {float(loss):>8.4f}  {time.time() - t0:>8.3f}s")
    total_time = time.time() - t_start

    # Validation loss on held-out text (fixed batches)
    val_rng = np.random.default_rng(123)
    val_batcher = make_batches(val_arr, SEQ_LEN, 64, val_rng)
    val_loss = float(np.mean([loss_fn(params, *next(val_batcher)) for _ in range(10)]))

    ckpt = f"checkpoint_{mixer}.npz"
    np.savez(ckpt, **{k: np.array(v) for k, v in params.items()}, vocab_size=vocab_size)
    with open(VOCAB_FILE, "w") as f:
        json.dump({"stoi": stoi, "itos": {int(k): v for k, v in itos.items()}}, f)

    result = {
        "mixer": mixer,
        "n_params": n_params,
        "train_loss": history,
        "val_loss": val_loss,
        "seconds": total_time,
    }
    with open(f"loss_{mixer}.json", "w") as f:
        json.dump(result, f)

    if verbose:
        print(f"\n✓ {ckpt}  |  val loss {val_loss:.4f}  |  {total_time:.1f}s total")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", choices=["attn", "fourier"], default="attn")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    a = ap.parse_args()
    train(a.mixer, a.epochs, a.batch)
