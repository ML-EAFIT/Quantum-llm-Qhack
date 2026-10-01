# Quantum Shakespeare in IBM Qiskit

This is the PennyLane quantum transformer ported to IBM Qiskit. It can run on the Qiskit Aer simulator, on a noisy local copy of a real IBM chip, or on a real IBM quantum computer.

| File | What it does |
|---|---|
| `quantum_layer.py` | The quantum Fourier mixer circuit in Qiskit (RY → RZ → QFT → ⟨Z⟩), the three backends, and parameter-shift gradients wired into PyTorch |
| `model.py` | The transformer: embeddings → quantum block → feed-forward → LM head. It uses the same parameter names, shapes and seed-42 initialisation as the PennyLane version |
| `train.py` | Trains on Tiny Shakespeare and saves `checkpoint.npz` and `vocab.json` |
| `inference.py` | Generates text. With `--backend ibm` it runs on a real QPU |
| `estimate_cost.py` | Works out the IBM QPU minutes and dollar cost of any run |
| `data.py` | Downloads the data, builds the character vocabulary and makes batches |
| `checkpoint.npz`, `vocab.json` | A model already trained here (300 steps, about 9 minutes on the simulator) |

## Install

```bash
pip install -r requirements.txt
```

## Run it

```bash
# 1. Train on the simulator (free, about 10 min on a laptop)
python train.py

# 2. Generate text on the simulator
python inference.py --prompt "ROMEO:" --tokens 200 --topn --circuit

# 3. Free dress rehearsal on a local copy of IBM Kingston, with real noise and shots
python inference.py --backend fake --prompt "ROMEO:"

# 4. Run on a REAL IBM quantum computer (1 job, about 1 QPU-minute)
python inference.py --backend ibm --prompt "ROMEO:"
python inference.py --backend ibm --device ibm_kingston --shots 1024   # cheaper

# How much would it cost?
python estimate_cost.py
```

### One-time IBM account setup (needed only for `--backend ibm`)

1. Create a free account at <https://quantum.cloud.ibm.com> (Open Plan: 10 free QPU-minutes per month).
2. Copy your **API key** and your **instance CRN** from the dashboard.
3. Save them once:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_API_KEY",
    instance="YOUR_INSTANCE_CRN",
    set_as_default=True,
)
```

## How much does it cost?

These are IBM list prices from October 2026: Open Plan is free for 10 min/month, Pay-As-You-Go is $96/min billed per second, Flex is $72/min with a 400-min/yr minimum, and Premium is $48/min with a 5,200-min/yr minimum. QPU time is calculated with IBM's own formula: 2 s per job + (rep_delay + circuit duration) × circuits × shots. The circuit duration is measured by compiling the circuit for IBM Kingston (Heron): 18 two-qubit gates, 4.3 µs, plus a 250 µs reset between shots.

| What you do | QPU time | Cost |
|---|---|---|
| Train on the simulator | 0 | **$0** |
| Run the trained model on a real IBM QPU (4096 shots) | 1.1 min | **Free** on Open Plan, or **$105** Pay-As-You-Go |
| Same, with 1024 shots | 0.3 min | Free, or $29 |
| Train on a real IBM QPU (300 steps, 4096 shots) | 25,189 min (420 h) | **$2.4 million** Pay-As-You-Go, $1.8M Flex, $1.2M Premium |
| Train on a real IBM QPU with only 64 shots (very noisy) | 403 min | $38,683 |

**Recommendation:** train on the simulator for free, then run inference on real IBM hardware. That fits inside the free Open Plan. `train.py --backend ibm` will not start unless you add `--i-understand-the-cost`.

### Why training on hardware is so expensive

On a QPU, gradients have to come from the parameter-shift rule. Each distinct token input needs 1 + 2×8 = **17 circuits**. One step has about 284 distinct (character, position) inputs, so it needs about 4,800 circuits. At 4,096 shots each, that is about 20 million shots, or 84 QPU-minutes **per step**, and there are 300 steps. These numbers match what the simulator actually ran: 1,448,995 circuits over the 300 steps.

### Why inference is cheap

The quantum block runs on each token on its own, and the next character is read from the last position. The quantum work behind every prediction is therefore "the circuit for (last character, position 15)", and there are only 61 such circuits. `inference.py` runs them all in **one job** and then generates as much text as you want. Re-running all 16 positions for every character, as the PennyLane script does, would cost 37 min ≈ $3,584 for 120 characters on Pay-As-You-Go.

## Checks that were run

- **Circuit:** the Qiskit circuit gives the same ⟨Z⟩ values as the PennyLane circuit (max difference 2e-15). PennyLane counts qubits in the opposite order, so the QFT gets the qubits reversed.
- **Gradients:** parameter-shift gradients match PennyLane autograd (max difference 9e-16).
- **Whole model:** with the same initial weights and the same batch, the loss matches the original `train.py` to 9e-16 and every parameter gradient matches to 5e-17.
- **Inference shortcut:** the one-job table matches the original `inference.py` full 16-token forward pass to 4e-15, using the same checkpoint.
- **Checkpoints:** files can be swapped both ways between the Qiskit and PennyLane versions.

## Things to know about this model

1. **It does not mix information between tokens.** Each token's circuit sees only that token's own 4 features. In FNet, the Fourier transform runs across the sequence; here it does not. As a result, the next-character prediction depends only on the current character, which makes this a bigram model. That is why the output looks like Shakespeare-shaped letter soup (loss drops from 4.18 to about 2.4). To get real context, tokens have to interact, for example through a mixer across positions or classical attention.
2. **There is no quantum speed-up here.** A laptop simulates 4 qubits exactly and instantly. Real hardware only adds noise and cost. The value of this project is learning how to build and run a hybrid quantum-classical model on IBM hardware.
3. Error mitigation (`--resilience 1`, IBM's default) adds a little QPU time on top of these estimates. `--resilience 0` is the cheapest setting.
4. `--backend fake` is good for an inference rehearsal (about 20 s). Training with it is slow: noisy simulation takes about 30 s per tiny step.

Pricing source: <https://www.ibm.com/quantum/products>. Usage formula: <https://quantum.cloud.ibm.com/docs/guides/estimate-job-run-time>.
