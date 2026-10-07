"""
inference.py — Generate Shakespeare with the classical baselines
================================================================
Usage
  python inference.py --mixer attn
  python inference.py --mixer fourier --prompt "To be or not" --tokens 200
"""

import argparse, json
import numpy as np

SEQ_LEN = 16
D_MODEL = 8
N_QUBITS = D_MODEL // 2
VOCAB_FILE = "vocab.json"

_j, _k = np.meshgrid(np.arange(N_QUBITS), np.arange(N_QUBITS), indexing="ij")
DFT_C = np.cos(2 * np.pi * _j * _k / N_QUBITS) / np.sqrt(N_QUBITS)
DFT_S = np.sin(2 * np.pi * _j * _k / N_QUBITS) / np.sqrt(N_QUBITS)


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def relu(x):
    return np.maximum(x, 0)


def softmax(logits, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def fourier_mix(x, phase):
    x_half, r_half = x[:, :N_QUBITS], x[:, N_QUBITS:]
    re = (x_half * np.cos(phase)) @ DFT_C + (x_half * np.sin(phase)) @ DFT_S
    return np.concatenate([np.tanh(re), r_half], axis=1)


def attention_mix(x, p):
    T = x.shape[0]
    q, k, v = x @ p["wq"], x @ p["wk"], x @ p["wv"]
    scores = (q @ k.T) / np.sqrt(D_MODEL)
    scores = np.where(np.tril(np.ones((T, T))) > 0, scores, -1e9)
    return (softmax(scores) @ v) @ p["wo"]


def forward(token_ids, params):
    T = len(token_ids)
    x = params["tok_emb"][token_ids] + params["pos_emb"][:T]

    residual = x
    x_norm = layer_norm(x, params["ln1_g"], params["ln1_b"])
    if "wq" in params:
        mixed = attention_mix(x_norm, params)
    else:
        mixed = fourier_mix(x_norm, params["f_phase"])
    x = residual + mixed

    residual = x
    x_norm = layer_norm(x, params["ln2_g"], params["ln2_b"])
    h = relu(x_norm @ params["ff_w1"] + params["ff_b1"])
    x = residual + h @ params["ff_w2"] + params["ff_b2"]

    logits = x @ params["lm_w"] + params["lm_b"]
    return logits[-1]


def load_checkpoint(ckpt_path, vocab_path=VOCAB_FILE):
    data = np.load(ckpt_path, allow_pickle=False)
    with open(vocab_path) as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    params = {k: data[k] for k in data.files if k != "vocab_size"}
    kind = "attn" if "wq" in params else "fourier"
    print(f"✓ Loaded {ckpt_path}  (mixer={kind}, vocab={len(stoi)})")
    return params, stoi, itos


def sample_top_k(probs, k=10):
    idx = np.argsort(probs)[-k:]
    p = probs[idx] / probs[idx].sum()
    return int(np.random.choice(idx, p=p))


def generate(
    prompt, params, stoi, itos, n_tokens=120, temperature=0.9, top_k=10, seed=0
):
    np.random.seed(seed)
    context = [stoi.get(c, stoi.get(" ", 0)) for c in prompt] or [0]
    generated = list(prompt)
    for _ in range(n_tokens):
        ctx = context[-SEQ_LEN:]
        if len(ctx) < SEQ_LEN:
            ctx = [ctx[0]] * (SEQ_LEN - len(ctx)) + ctx
        logits = forward(np.array(ctx, dtype=np.int32), params)
        probs = softmax(logits, temperature)
        nxt = sample_top_k(probs, k=min(top_k, len(stoi)))
        context.append(nxt)
        generated.append(itos[nxt])
    return "".join(generated)


def top_predictions(prompt, params, stoi, itos, n=5, temperature=0.8):
    context = [stoi.get(c, stoi.get(" ", 0)) for c in prompt] or [0]
    ctx = context[-SEQ_LEN:]
    if len(ctx) < SEQ_LEN:
        ctx = [ctx[0]] * (SEQ_LEN - len(ctx)) + ctx
    probs = softmax(forward(np.array(ctx, dtype=np.int32), params), temperature)
    print(f"\nTop-{n} predictions after '{prompt}':")
    for idx in np.argsort(probs)[-n:][::-1]:
        print(f"  {repr(itos[int(idx)]):>6}  {probs[idx]:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixer", choices=["attn", "fourier"], default="attn")
    p.add_argument("--prompt", default="To be")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topn", action="store_true")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--vocab", default=VOCAB_FILE)
    a = p.parse_args()

    params, stoi, itos = load_checkpoint(a.ckpt or f"checkpoint_{a.mixer}.npz", a.vocab)
    if a.topn:
        top_predictions(a.prompt, params, stoi, itos)

    print(f"\nPrompt: '{a.prompt}'\n" + "─" * 60)
    print(generate(a.prompt, params, stoi, itos, a.tokens, a.temp, a.top_k, a.seed))
    print("─" * 60)


if __name__ == "__main__":
    main()
