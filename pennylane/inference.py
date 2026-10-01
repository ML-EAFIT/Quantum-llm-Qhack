"""
inference.py — Generate Shakespeare with the Quantum Transformer
================================================================
Loads the checkpoint produced by train.py and auto-regressively
samples new text token by token.

Usage
  python inference.py
  python inference.py --prompt "To be or not" --tokens 200 --temp 0.8
"""

import argparse, json
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

# ──────────────────────────────────────────────────────────
# Defaults (must match train.py)
# ──────────────────────────────────────────────────────────
SEQ_LEN = 16
D_MODEL = 8
N_QUBITS = D_MODEL // 2
FF_DIM = 32
CKPT_FILE = "checkpoint.npz"
VOCAB_FILE = "vocab.json"

# ──────────────────────────────────────────────────────────
# Quantum circuit  (identical to train.py — no grad needed)
# ──────────────────────────────────────────────────────────

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def quantum_fourier_mixer(x_vec, rot_weights):
    qml.AngleEmbedding(x_vec, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS):
        qml.RZ(rot_weights[i], wires=i)
    qml.QFT(wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def quantum_mix_sequence(seq_emb, rot_weights):
    T, _ = seq_emb.shape
    rows = []
    for t in range(T):
        x_half = seq_emb[t, :N_QUBITS]
        r_half = seq_emb[t, N_QUBITS:]
        q_out = np.array(quantum_fourier_mixer(x_half, rot_weights))
        rows.append(np.concatenate([q_out, r_half]))
    return np.stack(rows)  # (T, D_MODEL)


# ──────────────────────────────────────────────────────────
# Model (inference-only, plain numpy — no autograd overhead)
# ──────────────────────────────────────────────────────────


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def relu(x):
    return np.maximum(x, 0)


def softmax(logits, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def forward(token_ids, params):
    T = len(token_ids)
    pos_emb = params["pos_emb"]
    # Support checkpoints trained with any SEQ_LEN
    pos = (
        pos_emb[:T]
        if T <= len(pos_emb)
        else np.tile(pos_emb, (T // len(pos_emb) + 1, 1))[:T]
    )
    x = params["tok_emb"][token_ids] + pos

    # Quantum attention block
    residual = x
    x_norm = layer_norm(x, params["ln1_g"], params["ln1_b"])
    x_mixed = quantum_mix_sequence(x_norm, params["q_rot"])
    x = residual + x_mixed

    # Feed-forward block
    residual = x
    x_norm = layer_norm(x, params["ln2_g"], params["ln2_b"])
    h = relu(x_norm @ params["ff_w1"] + params["ff_b1"])
    x = residual + h @ params["ff_w2"] + params["ff_b2"]

    # LM head — return raw logits for last token only
    logits = x @ params["lm_w"] + params["lm_b"]  # (T, vocab)
    return logits[-1]  # (vocab,)


# ──────────────────────────────────────────────────────────
# Checkpoint loader
# ──────────────────────────────────────────────────────────


def load_checkpoint(ckpt_path, vocab_path):
    data = np.load(ckpt_path, allow_pickle=False)
    vocab = json.load(open(vocab_path))
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    params = {k: data[k] for k in data if k != "vocab_size"}
    print(f"✓ Loaded checkpoint: {ckpt_path}")
    print(
        f"  vocab size = {len(stoi)}, "
        f"d_model = {params['tok_emb'].shape[1]}, "
        f"qubits = {N_QUBITS}"
    )
    return params, stoi, itos


# ──────────────────────────────────────────────────────────
# Sampling strategies
# ──────────────────────────────────────────────────────────


def sample_top_k(probs, k=10):
    """Sample from the top-k most likely tokens."""
    top_k_idx = np.argsort(probs)[-k:]
    top_k_prob = probs[top_k_idx]
    top_k_prob = top_k_prob / top_k_prob.sum()
    return int(np.random.choice(top_k_idx, p=top_k_prob))


def generate(
    prompt, params, stoi, itos, n_tokens=120, temperature=0.9, top_k=10, seed=0
):
    """
    prompt     : string seed text (must only contain known chars)
    n_tokens   : how many new characters to generate
    temperature: >1 more random, <1 more greedy
    top_k      : sample from top-k logits
    """
    np.random.seed(seed)
    vocab_size = len(stoi)

    # Encode prompt; fall back to space for unknown chars
    context = [stoi.get(c, stoi.get(" ", 0)) for c in prompt]
    if not context:
        context = [0]

    generated = list(prompt)

    for step in range(n_tokens):
        # Keep last SEQ_LEN tokens as the context window
        ctx = context[-SEQ_LEN:]
        # Pad with first token if shorter than SEQ_LEN
        if len(ctx) < SEQ_LEN:
            ctx = [ctx[0]] * (SEQ_LEN - len(ctx)) + ctx
        ids = np.array(ctx, dtype=np.int32)
        logits = forward(ids, params)  # (vocab,)
        probs = softmax(logits, temperature)
        next_id = sample_top_k(probs, k=min(top_k, vocab_size))
        context.append(next_id)
        generated.append(itos[next_id])

    return "".join(generated)


# ──────────────────────────────────────────────────────────
# Introspection helpers
# ──────────────────────────────────────────────────────────


def print_circuit_info():
    """Print a summary of the quantum mixer circuit."""
    print("\n── Quantum Fourier Mixer Circuit ──")
    dummy_x = pnp.zeros(N_QUBITS)
    dummy_rot = pnp.zeros(N_QUBITS)
    print(qml.draw(quantum_fourier_mixer, max_length=80)(dummy_x, dummy_rot))
    print()


def top_predictions(prompt, params, stoi, itos, n=5, temperature=0.8):
    """Show the top-n next-token predictions for a given prompt."""
    context = [stoi.get(c, stoi.get(" ", 0)) for c in prompt]
    if not context:
        context = [0]
    ctx = context[-SEQ_LEN:]
    if len(ctx) < SEQ_LEN:
        ctx = [ctx[0]] * (SEQ_LEN - len(ctx)) + ctx
    ids = np.array(ctx, dtype=np.int32)
    logits = forward(ids, params)
    probs = softmax(logits, temperature)
    top_n = np.argsort(probs)[-n:][::-1]
    print(f"\nTop-{n} predictions after '{prompt}':")
    print(f"  {'Char':>6}  {'Prob':>8}")
    print("  " + "-" * 18)
    for idx in top_n:
        char = itos[int(idx)]
        print(f"  {repr(char):>6}  {probs[idx]:>8.4f}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Shakespeare with the Quantum Transformer"
    )
    p.add_argument("--prompt", default="To be", help="Seed text (default: 'To be')")
    p.add_argument(
        "--tokens",
        type=int,
        default=120,
        help="Number of new characters to generate (default: 120)",
    )
    p.add_argument(
        "--temp", type=float, default=0.9, help="Sampling temperature (default: 0.9)"
    )
    p.add_argument("--top_k", type=int, default=10, help="Top-k sampling (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--circuit", action="store_true", help="Print the quantum circuit diagram"
    )
    p.add_argument(
        "--topn", action="store_true", help="Show top-5 next-token predictions"
    )
    p.add_argument(
        "--ckpt", default=CKPT_FILE, help=f"Checkpoint path (default: {CKPT_FILE})"
    )
    p.add_argument(
        "--vocab", default=VOCAB_FILE, help=f"Vocab path (default: {VOCAB_FILE})"
    )
    return p.parse_args()


def main():
    args = parse_args()
    params, stoi, itos = load_checkpoint(args.ckpt, args.vocab)

    if args.circuit:
        print_circuit_info()

    if args.topn:
        top_predictions(args.prompt, params, stoi, itos)

    print(
        f"\n── Generating {args.tokens} tokens "
        f"(T={args.temp}, top-k={args.top_k}) ──\n"
    )
    print(f"Prompt: '{args.prompt}'\n")
    print("─" * 60)
    result = generate(
        args.prompt,
        params,
        stoi,
        itos,
        n_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(result)
    print("\n" + "─" * 60)


if __name__ == "__main__":
    main()
