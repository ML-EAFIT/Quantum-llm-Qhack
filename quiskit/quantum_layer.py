"""
quantum_layer.py — the Quantum Fourier Mixer, written in IBM Qiskit
====================================================================
Same circuit as the PennyLane version, gate for gate:

    RY(x_i) on every qubit   (AngleEmbedding, rotation="Y")
    RZ(w_i) on every qubit   (learnable phases)
    QFT on all qubits
    measure <Z_i> on every qubit

Three places to run it (chosen with --backend):

  aer   exact simulation with Qiskit Aer .......................... free
  fake  Qiskit Aer + the noise model of a real IBM Heron chip,
        with shots — a free "dress rehearsal" of the hardware ...... free
  ibm   a real IBM quantum computer via qiskit-ibm-runtime ....... costs QPU time

Gradients use the parameter-shift rule, which is how gradients are taken
on real quantum hardware (you cannot back-propagate through a QPU):

    d<Z>/dθ = ( <Z>(θ + π/2) − <Z>(θ − π/2) ) / 2

With 4 qubits there are 8 circuit parameters (4 inputs + 4 weights), so
one gradient needs 1 + 2·8 = 17 circuits per distinct token input.
"""

import math
import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import SparsePauliOp

SHIFT = np.pi / 2
FAKE_DEVICES = ("kingston", "fez", "marrakesh", "torino", "aachen", "boston", "pittsburgh", "miami")


# ──────────────────────────────────────────────────────────
# 1. The circuit
# ──────────────────────────────────────────────────────────


def build_circuit(n_qubits):
    """Return the parameterised Qiskit circuit (x = inputs, w = weights)."""
    x = ParameterVector("x", n_qubits)
    w = ParameterVector("w", n_qubits)
    qc = QuantumCircuit(n_qubits, name="quantum_fourier_mixer")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    for i in range(n_qubits):
        qc.rz(w[i], i)
    # PennyLane treats wire 0 as the most-significant bit; Qiskit treats
    # qubit 0 as the least-significant. Handing the QFT the qubits in
    # reverse order makes this exactly qml.QFT(wires=range(n)), so results
    # (and checkpoints) match the PennyLane version.
    qc.append(QFTGate(n_qubits), list(reversed(range(n_qubits))))
    return qc


def z_observables(n_qubits):
    """<Z_i> for every qubit i (all commute, so one measurement setting)."""
    return [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], n_qubits) for i in range(n_qubits)]


def get_fake_backend(name="kingston"):
    """A local copy of a real IBM chip: same qubit layout, gate set and noise."""
    from qiskit_ibm_runtime import fake_provider

    cls = getattr(fake_provider, "Fake" + name.strip().lower().replace("ibm_", "").capitalize(), None)
    if cls is None:
        raise ValueError(f"Unknown fake device '{name}'. Try one of: {', '.join(FAKE_DEVICES)}")
    return cls()


def get_ibm_backend(device=None, n_qubits=4):
    """A real IBM quantum computer (needs a saved IBM Quantum account)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    if device:
        return service.backend(device)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)


# ──────────────────────────────────────────────────────────
# 2. Running the circuit on a backend
# ──────────────────────────────────────────────────────────


class QuantumMixer:
    """
    Evaluates the mixer circuit for many parameter sets at once, using a
    Qiskit Estimator primitive. Everything is sent as one broadcasted PUB
    (split into a few jobs only if it would exceed the per-job limit), which
    keeps per-job overhead — and the bill — as small as possible.
    """

    def __init__(
        self,
        n_qubits,
        backend="aer",
        shots=4096,
        device=None,
        resilience_level=1,
        max_executions_per_job=5_000_000,
    ):
        self.n = n_qubits
        self.kind = backend
        self.shots = None if backend == "aer" else int(shots)
        self.max_exec = int(max_executions_per_job)
        self.logical_circuit = build_circuit(n_qubits)
        obs = z_observables(n_qubits)

        # usage counters (handy for checking the cost estimate)
        self.circuits_run = 0
        self.jobs_run = 0
        self.qpu_seconds = 0.0

        if backend == "aer":
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator

            self.backend_name = "qiskit-aer (exact statevector)"
            self.circuit = transpile(
                self.logical_circuit,
                basis_gates=["ry", "rz", "h", "cp", "cx", "swap", "u", "sx", "x"],
                optimization_level=0,
            )
            self.observables = obs
            self.estimator = AerEstimator(options={"default_precision": 0.0})
        elif backend in ("fake", "ibm"):
            from qiskit.transpiler import generate_preset_pass_manager
            from qiskit_ibm_runtime import EstimatorV2

            hw = get_fake_backend(device or "kingston") if backend == "fake" else get_ibm_backend(device, n_qubits)
            self.hw_backend = hw
            self.backend_name = hw.name + (" (local noisy simulation)" if backend == "fake" else " (REAL QPU)")
            pm = generate_preset_pass_manager(optimization_level=3, backend=hw, seed_transpiler=42)
            self.circuit = pm.run(self.logical_circuit)
            self.observables = [o.apply_layout(self.circuit.layout) for o in obs]
            self.estimator = EstimatorV2(mode=hw)
            self.estimator.options.default_shots = self.shots
            if backend == "ibm":
                self.estimator.options.resilience_level = resilience_level
        else:
            raise ValueError("backend must be 'aer', 'fake' or 'ibm'")

        # Qiskit orders circuit parameters by name (w[...] before x[...]).
        # Our arrays are laid out [x_0..x_{n-1}, w_0..w_{n-1}]; build the map.
        self._perm = np.array(
            [(0 if p.vector.name == "x" else n_qubits) + p.index for p in self.circuit.parameters]
        )
        self._obs_column = [[o] for o in self.observables]  # shape (n, 1) → broadcasts

    # one row of P = [x_0..x_{n-1}, w_0..w_{n-1}]
    def run_params(self, P):
        P = np.asarray(P, dtype=np.float64)
        N = P.shape[0]
        out = np.empty((N, self.n))
        values = P[:, self._perm]
        chunk = N if self.shots is None else max(1, self.max_exec // self.shots)
        for s in range(0, N, chunk):
            block = values[s : s + chunk]
            pub = (self.circuit, self._obs_column, block[None, :, :])  # evs → (n, len(block))
            job = self.estimator.run([pub])
            out[s : s + len(block)] = job.result()[0].data.evs.T
            self.jobs_run += 1
            if self.kind == "ibm":
                try:
                    self.qpu_seconds += float(job.usage())
                except Exception:
                    pass
        self.circuits_run += N
        return out

    def expvals(self, X, w):
        """X: (N, n) inputs, w: (n,) weights → (N, n) values of <Z_i>."""
        X = np.asarray(X, dtype=np.float64)
        return self.run_params(np.concatenate([X, np.broadcast_to(w, X.shape)], axis=1))

    def draw(self):
        return self.logical_circuit.decompose(gates_to_decompose=["qft"]).draw(output="text", fold=100)

    def usage_report(self):
        msg = f"{self.circuits_run:,} circuits in {self.jobs_run} job(s) on {self.backend_name}"
        if self.shots:
            msg += f", {self.shots} shots each"
        if self.kind == "ibm" and self.qpu_seconds:
            msg += f" — QPU usage {self.qpu_seconds:.1f}s ≈ ${self.qpu_seconds / 60 * 96:,.2f} at $96/min"
        return msg


# ──────────────────────────────────────────────────────────
# 3. PyTorch autograd bridge (parameter-shift gradients)
# ──────────────────────────────────────────────────────────


class _QuantumMixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, mixer):
        n = mixer.n
        X = x.detach().cpu().double().numpy()
        W = w.detach().cpu().double().numpy()

        # A token's circuit input depends only on (character, position), so a
        # batch repeats many inputs. Run each distinct input once.
        uniq, inv = np.unique(X, axis=0, return_inverse=True)
        inv = inv.reshape(-1)

        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            out = mixer.expvals(uniq, W)
            return torch.as_tensor(out[inv], dtype=x.dtype)

        # Forward + all parameter shifts go out together in ONE batch.
        k = 2 * n
        base = np.concatenate([uniq, np.broadcast_to(W, uniq.shape)], axis=1)  # (U, k)
        shifts = np.concatenate([np.eye(k), -np.eye(k)]) * SHIFT  # (2k, k)
        allp = np.concatenate([base[:, None, :], base[:, None, :] + shifts[None]], axis=1)  # (U, 1+2k, k)
        res = mixer.run_params(allp.reshape(-1, k)).reshape(len(uniq), 1 + 2 * k, n)

        f0 = res[:, 0]
        jac = (res[:, 1 : 1 + k] - res[:, 1 + k :]) / 2.0  # (U, k params, n outputs)
        ctx.save_for_backward(torch.as_tensor(jac), torch.as_tensor(inv))
        ctx.n = n
        return torch.as_tensor(f0[inv], dtype=x.dtype)

    @staticmethod
    def backward(ctx, g):
        jac, inv = ctx.saved_tensors
        n = ctx.n
        g = g.double()
        gx = torch.einsum("ikn,in->ik", jac[inv][:, :n, :], g)
        G = torch.zeros(jac.shape[0], n, dtype=torch.float64).index_add_(0, inv, g)
        gw = torch.einsum("ukn,un->k", jac[:, n:, :], G)
        return gx.to(g.dtype), gw.to(g.dtype), None


def quantum_mix(x, w, mixer):
    """x: (N, n_qubits) tensor, w: (n_qubits,) tensor → (N, n_qubits) <Z>."""
    if not (torch.is_grad_enabled() and (x.requires_grad or w.requires_grad)):
        # inference: no gradient circuits, just one circuit per distinct input
        X = x.detach().cpu().double().numpy()
        uniq, inv = np.unique(X, axis=0, return_inverse=True)
        out = mixer.expvals(uniq, w.detach().cpu().double().numpy())
        return torch.as_tensor(out[inv.reshape(-1)], dtype=x.dtype)
    return _QuantumMixFn.apply(x, w, mixer)


def shots_note(shots):
    """Statistical error of one <Z> estimate with this many shots."""
    return f"±{1 / math.sqrt(shots):.3f}" if shots else "exact"
