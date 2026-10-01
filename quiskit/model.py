"""
model.py — the Quantum Transformer (PyTorch + Qiskit)
=====================================================
Same architecture, same parameter names and shapes, and the same random
initialisation (seed 42) as the PennyLane version, so checkpoint.npz files
are interchangeable between the two.

  - Classical token & positional embeddings (vocab → d_model)
  - Quantum block: Qiskit Fourier mixer on the first d_model/2 features
  - Classical feed-forward (ReLU)
  - Classical LM head → log-probabilities
"""

import numpy as np
import torch
import torch.nn as nn

from quantum_layer import quantum_mix

SEQ_LEN = 16
D_MODEL = 8
N_QUBITS = D_MODEL // 2
FF_DIM = 32

PARAM_NAMES = [
    "tok_emb", "pos_emb", "q_rot",
    "ff_w1", "ff_b1", "ff_w2", "ff_b2",
    "ln1_g", "ln1_b", "ln2_g", "ln2_b",
    "lm_w", "lm_b",
]


class QuantumShakespeare(nn.Module):
    def __init__(self, vocab_size, mixer, seed=42):
        super().__init__()
        self.mixer = mixer
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)

        def r(*shape):
            return nn.Parameter(torch.tensor(rng.normal(0, 0.1, shape), dtype=torch.float64))

        # (same draw order as the PennyLane init_params → identical start)
        self.tok_emb = r(vocab_size, D_MODEL)
        self.pos_emb = r(SEQ_LEN, D_MODEL)
        self.q_rot = r(N_QUBITS)
        self.ff_w1 = r(D_MODEL, FF_DIM)
        self.ff_b1 = r(FF_DIM)
        self.ff_w2 = r(FF_DIM, D_MODEL)
        self.ff_b2 = r(D_MODEL)
        self.ln1_g = nn.Parameter(torch.ones(D_MODEL, dtype=torch.float64))
        self.ln1_b = nn.Parameter(torch.zeros(D_MODEL, dtype=torch.float64))
        self.ln2_g = nn.Parameter(torch.ones(D_MODEL, dtype=torch.float64))
        self.ln2_b = nn.Parameter(torch.zeros(D_MODEL, dtype=torch.float64))
        self.lm_w = r(D_MODEL, vocab_size)
        self.lm_b = r(vocab_size)

    @staticmethod
    def layer_norm(x, g, b, eps=1e-5):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return g * (x - mean) / torch.sqrt(var + eps) + b

    def _pos(self, positions):
        # supports any context length (wraps like the PennyLane inference)
        return self.pos_emb[positions % SEQ_LEN]

    def forward(self, token_ids, positions=None):
        """
        token_ids : (B, T) long tensor
        positions : optional (B, T) or (T,) position indices (default 0..T-1)
        returns   : (B, T, vocab) log-probabilities
        """
        B, T = token_ids.shape
        if positions is None:
            positions = torch.arange(T)
        x = self.tok_emb[token_ids] + self._pos(positions)

        # ── Quantum block ────────────────────────────────
        residual = x
        xn = self.layer_norm(x, self.ln1_g, self.ln1_b)
        q = quantum_mix(xn[..., :N_QUBITS].reshape(-1, N_QUBITS), self.q_rot, self.mixer)
        x_mixed = torch.cat([q.reshape(B, T, N_QUBITS), xn[..., N_QUBITS:]], dim=-1)
        x = residual + x_mixed

        # ── Feed-forward block ───────────────────────────
        residual = x
        xn = self.layer_norm(x, self.ln2_g, self.ln2_b)
        h = torch.relu(xn @ self.ff_w1 + self.ff_b1)
        x = residual + h @ self.ff_w2 + self.ff_b2

        # ── LM head ──────────────────────────────────────
        logits = x @ self.lm_w + self.lm_b
        return torch.log_softmax(logits, dim=-1)

    # ── checkpoint I/O (same format as the PennyLane version) ──
    def save(self, path):
        d = {k: getattr(self, k).detach().numpy() for k in PARAM_NAMES}
        d["vocab_size"] = np.array(self.vocab_size)
        np.savez(path, **d)

    @classmethod
    def load(cls, path, mixer):
        data = np.load(path, allow_pickle=False)
        model = cls(int(data["vocab_size"]), mixer)
        with torch.no_grad():
            for k in PARAM_NAMES:
                getattr(model, k).copy_(torch.tensor(data[k], dtype=torch.float64))
        return model
