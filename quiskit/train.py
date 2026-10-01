"""
train.py — Quantum Transformer trained on Shakespeare (IBM Qiskit version)
==========================================================================
Port of the PennyLane train.py to Qiskit. Same model, same hyper-parameters,
same checkpoint format.

Backends
  --backend aer    exact Qiskit Aer simulation (free, default, recommended)
  --backend fake   Aer with the noise of a real IBM Heron chip + shots (free)
  --backend ibm    a REAL IBM quantum computer — very expensive for training,
                   see estimate_cost.py; needs --i-understand-the-cost

Usage
  python train.py
  python train.py --backend fake --shots 1024 --steps 50
  → saves checkpoint.npz (weights) + vocab.json
"""

import argparse
import json
import time

import numpy as np
import torch

from data import fetch_shakespeare, build_vocab, encode, make_batches, MAX_CHARS
from model import QuantumShakespeare, SEQ_LEN, D_MODEL, N_QUBITS
from quantum_layer import QuantumMixer, shots_note

BATCH = 32  # sequences per gradient step
STEPS = 300  # gradient steps (the PennyLane script called these "epochs")
LR = 0.02  # Adam learning rate
CKPT_FILE = "checkpoint.npz"
VOCAB_FILE = "vocab.json"


def parse_args():
    p = argparse.ArgumentParser(description="Train the quantum Shakespeare model with Qiskit")
    p.add_argument("--backend", choices=["aer", "fake", "ibm"], default="aer")
    p.add_argument("--device", default=None, help="fake: kingston|fez|torino…  ibm: e.g. ibm_kingston (default: least busy)")
    p.add_argument("--shots", type=int, default=4096, help="shots per circuit (fake/ibm only)")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--max-chars", type=int, default=MAX_CHARS)
    p.add_argument("--ckpt", default=CKPT_FILE)
    p.add_argument("--vocab", default=VOCAB_FILE)
    p.add_argument("--i-understand-the-cost", action="store_true",
                   help="required to train on real IBM hardware")
    return p.parse_args()


def cost_guard(args):
    """Refuse to start a real-hardware training run by accident."""
    from estimate_cost import estimate, money, PAYG

    r = estimate(args.shots, args.steps, args.batch, verbose=False)
    print(f"\n⚠  Training on real IBM hardware: ≈ {r['train_min']:,.0f} QPU-minutes "
          f"≈ {money(r['train_min'], PAYG)} at Pay-As-You-Go rates.")
    if not args.i_understand_the_cost:
        raise SystemExit("   Not starting. Train with --backend aer (free) and run inference on hardware instead,\n"
                         "   or add --i-understand-the-cost if you really mean it.")


def train(args):
    if args.backend == "ibm":
        cost_guard(args)

    torch.manual_seed(0)
    text = fetch_shakespeare(args.max_chars)
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}  |  Data: {len(text)} chars")

    mixer = QuantumMixer(N_QUBITS, backend=args.backend, shots=args.shots, device=args.device)
    print(f"Quantum backend: {mixer.backend_name}  "
          f"(shots: {mixer.shots or '—'}, <Z> precision {shots_note(mixer.shots)})")

    model = QuantumShakespeare(vocab_size, mixer)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    batcher = make_batches(encode(text, stoi), SEQ_LEN, args.batch)

    print(f"\nTraining {args.steps} steps  (seq={SEQ_LEN}, d={D_MODEL}, qubits={N_QUBITS}, batch={args.batch})\n")
    print(f"{'Step':>6}  {'Loss':>8}  {'Time/step':>9}  {'Circuits so far':>15}")
    print("-" * 46)

    t_start = time.time()
    for step in range(1, args.steps + 1):
        X, Y = next(batcher)
        t0 = time.time()
        log_p = model(torch.as_tensor(X))  # (B, T, vocab)
        loss = torch.nn.functional.nll_loss(log_p.reshape(-1, vocab_size), torch.as_tensor(Y).reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == 1:
            print(f"{step:>6}  {loss.item():>8.4f}  {time.time() - t0:>8.2f}s  {mixer.circuits_run:>15,}")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min — {mixer.usage_report()}")

    model.save(args.ckpt)
    with open(args.vocab, "w") as f:
        json.dump({"stoi": stoi, "itos": {int(k): v for k, v in itos.items()}}, f)
    print(f"✓ Checkpoint saved → {args.ckpt}")
    print(f"✓ Vocabulary saved → {args.vocab}")


if __name__ == "__main__":
    train(parse_args())
