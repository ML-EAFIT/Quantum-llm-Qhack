"""
inference.py — Generate Shakespeare with the Quantum Transformer (Qiskit)
=========================================================================
Loads checkpoint.npz (from train.py — this one or the PennyLane one) and
samples new text character by character.

Why inference is cheap on real hardware
  The quantum block runs on each token separately, and the prediction for
  the next character is read from the LAST position only. So the quantum
  part of every prediction is "the circuit for (last character, position 15)".
  There are only vocab_size such circuits, so we run them all in ONE batched
  job up front and then generate as much text as we like from those results.
  In exact simulation this gives the same output as re-running the full
  16-token forward pass every step (the test in README checks this).

Usage
  python inference.py
  python inference.py --prompt "To be or not" --tokens 200 --temp 0.8
  python inference.py --backend fake                 # noisy IBM-chip simulation
  python inference.py --backend ibm                  # REAL IBM QPU, 1 job, ~1 min
  python inference.py --backend ibm --device ibm_kingston --shots 1024
"""

import argparse
import json

import numpy as np
import torch

from model import QuantumShakespeare, SEQ_LEN, N_QUBITS
from quantum_layer import QuantumMixer, shots_note

CKPT_FILE = "checkpoint.npz"
VOCAB_FILE = "vocab.json"


def load_vocab(path):
    vocab = json.load(open(path))
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    return stoi, itos


@torch.no_grad()
def next_char_table(model):
    """
    (vocab, vocab) log-probabilities: row c = distribution of the next
    character when the last character of the context is c. One quantum job.
    """
    V = model.vocab_size
    ids = torch.arange(V).reshape(V, 1)
    pos = torch.full((V, 1), SEQ_LEN - 1)  # the context is always padded to 16
    return model(ids, positions=pos)[:, 0, :].numpy()


def encode_prompt(prompt, stoi):
    ctx = [stoi.get(c, stoi.get(" ", 0)) for c in prompt]
    return ctx or [0]


def softmax(logits, temperature=1.0):
    z = logits / max(temperature, 1e-8)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample_top_k(probs, k, rng):
    top = np.argsort(probs)[-k:]
    p = probs[top] / probs[top].sum()
    return int(rng.choice(top, p=p))


def generate(prompt, table, stoi, itos, n_tokens=120, temperature=0.9, top_k=10, seed=42):
    rng = np.random.default_rng(seed)
    context = encode_prompt(prompt, stoi)
    out = list(prompt)
    for _ in range(n_tokens):
        probs = softmax(table[context[-1]], temperature)
        nxt = sample_top_k(probs, min(top_k, len(stoi)), rng)
        context.append(nxt)
        out.append(itos[nxt])
    return "".join(out)


def top_predictions(prompt, table, stoi, itos, n=5, temperature=0.8):
    probs = softmax(table[encode_prompt(prompt, stoi)[-1]], temperature)
    print(f"\nTop-{n} predictions after '{prompt}':")
    print(f"  {'Char':>6}  {'Prob':>8}")
    print("  " + "-" * 18)
    for idx in np.argsort(probs)[-n:][::-1]:
        print(f"  {repr(itos[int(idx)]):>6}  {probs[idx]:>8.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate Shakespeare with the Qiskit quantum transformer")
    p.add_argument("--prompt", default="To be")
    p.add_argument("--tokens", type=int, default=120, help="new characters to generate")
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["aer", "fake", "ibm"], default="aer")
    p.add_argument("--device", default=None, help="fake: kingston|fez|torino…  ibm: e.g. ibm_kingston")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--resilience", type=int, default=1, choices=[0, 1, 2],
                   help="ibm only: error-mitigation level (0 = cheapest, 1 = IBM default)")
    p.add_argument("--circuit", action="store_true", help="print the quantum circuit")
    p.add_argument("--topn", action="store_true", help="show top-5 next-character predictions")
    p.add_argument("--ckpt", default=CKPT_FILE)
    p.add_argument("--vocab", default=VOCAB_FILE)
    return p.parse_args()


def main():
    args = parse_args()
    stoi, itos = load_vocab(args.vocab)
    mixer = QuantumMixer(N_QUBITS, backend=args.backend, shots=args.shots, device=args.device,
                         resilience_level=args.resilience)
    model = QuantumShakespeare.load(args.ckpt, mixer)
    print(f"✓ Loaded {args.ckpt}: vocab={model.vocab_size}, qubits={N_QUBITS}")
    print(f"  Quantum backend: {mixer.backend_name}  (<Z> precision {shots_note(mixer.shots)})")

    if args.circuit:
        print("\n── Quantum Fourier Mixer Circuit ──")
        print(mixer.draw())

    table = next_char_table(model)
    print(f"  Quantum part: {mixer.usage_report()}")

    if args.topn:
        top_predictions(args.prompt, table, stoi, itos)

    print(f"\n── Generating {args.tokens} characters (T={args.temp}, top-k={args.top_k}) ──\n")
    print(f"Prompt: '{args.prompt}'\n")
    print("─" * 60)
    print(generate(args.prompt, table, stoi, itos, args.tokens, args.temp, args.top_k, args.seed))
    print("─" * 60)


if __name__ == "__main__":
    main()
