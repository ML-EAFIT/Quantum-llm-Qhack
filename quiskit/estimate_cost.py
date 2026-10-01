"""
estimate_cost.py — How much would this cost on a real IBM quantum computer?
===========================================================================
Uses IBM's own usage formula (docs: "Estimate usage before submitting a job"):

    QPU seconds = 2 s per job  +  (rep_delay + circuit_duration) × circuits × shots

circuit_duration is measured by compiling the circuit for a real IBM Heron
chip (a local "fake" copy of it, so no account is needed); rep_delay is that
chip's default (250 µs). Prices are IBM's list prices (ibm.com/quantum/products,
checked Oct 2026):

    Open Plan        free, up to 10 QPU-minutes per month
    Pay-As-You-Go    $96 / minute, billed per second, no minimum
    Flex             $72 / minute, minimum 400 minutes per year  ($28,800)
    Premium          $48 / minute, minimum 5,200 minutes per year ($249,600)

Usage
  python estimate_cost.py
  python estimate_cost.py --shots 1024 --steps 300 --batch 32
"""

import argparse
import math
import numpy as np

from data import fetch_shakespeare, build_vocab, encode, make_batches
from model import SEQ_LEN, N_QUBITS
from quantum_layer import build_circuit, get_fake_backend

PER_JOB_OVERHEAD_S = 2.0
FREE_MINUTES_PER_MONTH = 10
PAYG, FLEX, PREMIUM = 96.0, 72.0, 48.0
CIRCUITS_PER_GRADIENT = 1 + 2 * (2 * N_QUBITS)  # forward + ±π/2 shift of 8 params = 17


def circuit_timing(device="kingston"):
    """Duration of one shot of our circuit on a real IBM chip, in seconds."""
    from qiskit.transpiler import generate_preset_pass_manager

    backend = get_fake_backend(device)
    qc = build_circuit(N_QUBITS)
    qc.measure_all()
    isa = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42).run(qc)
    duration = isa.estimate_duration(backend.target, unit="s")
    rep_delay = getattr(backend, "default_rep_delay", None) or 250e-6
    ops = isa.count_ops()
    two_qubit = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
    return backend.name, float(duration), float(rep_delay), two_qubit


def qpu_seconds(n_circuits, shots, per_shot_s, max_exec=5_000_000):
    """IBM's formula, splitting into jobs the same way quantum_layer.py does."""
    per_job = max(1, max_exec // shots)
    jobs = math.ceil(n_circuits / per_job)
    return jobs * PER_JOB_OVERHEAD_S + n_circuits * shots * per_shot_s, jobs


def avg_unique_inputs(tokens, batch, samples=300):
    """Average number of distinct (character, position) pairs in one batch."""
    gen = make_batches(tokens, SEQ_LEN, batch, seed=123)
    pos = np.arange(SEQ_LEN)
    counts = []
    for _ in range(samples):
        X, _ = next(gen)
        counts.append(len(set(zip(X.ravel().tolist(), np.tile(pos, batch).tolist()))))
    return float(np.mean(counts))


def money(minutes, rate):
    return f"${minutes * rate:,.0f}" if minutes * rate >= 10 else f"${minutes * rate:,.2f}"


def estimate(shots=4096, steps=300, batch=32, chars=120, device="kingston", verbose=True):
    text = fetch_shakespeare()
    stoi, _ = build_vocab(text)
    V = len(stoi)
    tokens = encode(text, stoi)

    name, dur, rep, n2q = circuit_timing(device)
    per_shot = dur + rep
    U = avg_unique_inputs(tokens, batch)

    # Training on the QPU: every step needs forward + parameter-shift circuits
    step_circuits = U * CIRCUITS_PER_GRADIENT
    step_s, step_jobs = qpu_seconds(step_circuits, shots, per_shot)
    train_min = steps * step_s / 60

    # Inference, smart: one job with one circuit per character in the vocab
    inf_s, _ = qpu_seconds(V, shots, per_shot)
    inf_min = inf_s / 60

    # Inference, naive (like the PennyLane script): 16 circuits per new
    # character, one job per character
    naive_s = chars * qpu_seconds(SEQ_LEN, shots, per_shot)[0]
    naive_min = naive_s / 60

    r = dict(V=V, device=name, dur=dur, rep=rep, n2q=n2q, U=U, step_circuits=step_circuits,
             step_s=step_s, step_jobs=step_jobs, train_min=train_min, inf_min=inf_min,
             naive_min=naive_min, shots=shots, steps=steps, batch=batch, chars=chars)
    if verbose:
        report(r)
    return r


def report(r):
    print("\n══ What one shot costs in time ══════════════════════════════")
    print(f"  Device model         : {r['device']} (IBM Heron, 156 qubits)")
    print(f"  Compiled circuit     : {r['n2q']} two-qubit gates, {r['dur'] * 1e6:.1f} µs incl. readout")
    print(f"  Reset between shots  : {r['rep'] * 1e6:.0f} µs (rep_delay)")
    print(f"  Shots per circuit    : {r['shots']}  (error of each <Z> ≈ ±{1 / math.sqrt(r['shots']):.3f})")

    print("\n══ 1. Train on the simulator (recommended) ══════════════════")
    print("  QPU time 0 min  →  $0   (4 qubits simulate instantly on a laptop)")

    print("\n══ 2. Run the trained model on a REAL IBM quantum computer ══")
    print(f"  Vocabulary: {r['V']} characters → {r['V']} circuits, 1 job, any amount of text")
    print(f"  QPU time {r['inf_min']:.2f} min  →  Open Plan: FREE (fits in 10 free min/month)"
          f"  |  Pay-As-You-Go: {money(r['inf_min'], PAYG)}")
    print(f"  (Naive way, re-running 16 circuits per character like the PennyLane script:")
    print(f"   {r['chars']} chars = {r['naive_min']:.1f} min = {money(r['naive_min'], PAYG)} Pay-As-You-Go)")

    print("\n══ 3. TRAIN on a REAL IBM quantum computer ══════════════════")
    print(f"  Per step: ~{r['U']:.0f} distinct token inputs × {CIRCUITS_PER_GRADIENT} circuits"
          f" (parameter shift) = {r['step_circuits']:,.0f} circuits, {r['step_jobs']} job(s)")
    print(f"  Per step QPU time: {r['step_s'] / 60:.1f} min  ({money(r['step_s'] / 60, PAYG)} Pay-As-You-Go)")
    tm = r["train_min"]
    print(f"  {r['steps']} steps: {tm:,.0f} QPU-minutes = {tm / 60:,.0f} hours of QPU time")
    print(f"    Pay-As-You-Go $96/min : {money(tm, PAYG)}")
    print(f"    Flex          $72/min : {money(max(tm, 400), FLEX)}")
    print(f"    Premium       $48/min : {money(max(tm, 5200), PREMIUM)}")
    print(f"    Open Plan (free)      : would take {tm / FREE_MINUTES_PER_MONTH / 12:,.0f} years of free minutes")
    print()


def main():
    p = argparse.ArgumentParser(description="Estimate IBM Quantum cost for the quantum Shakespeare model")
    p.add_argument("--shots", type=int, default=4096, help="shots per circuit (IBM default precision ≈ 4096)")
    p.add_argument("--steps", type=int, default=300, help="training steps")
    p.add_argument("--batch", type=int, default=32, help="sequences per step")
    p.add_argument("--chars", type=int, default=120, help="characters generated (naive comparison)")
    p.add_argument("--device", default="kingston", help="IBM chip model to time the circuit on")
    a = p.parse_args()
    estimate(a.shots, a.steps, a.batch, a.chars, a.device)

    print("── Training cost vs shots (Pay-As-You-Go) ──")
    for s in (4096, 1024, 256, 64):
        r = estimate(s, a.steps, a.batch, a.chars, a.device, verbose=False)
        print(f"  {s:>5} shots (±{1 / math.sqrt(s):.3f}): train {r['train_min']:>9,.0f} min = "
              f"{money(r['train_min'], PAYG):>12}   |   inference {r['inf_min']:.2f} min = {money(r['inf_min'], PAYG)}")
    print()


if __name__ == "__main__":
    main()
