"""data.py — Tiny Shakespeare download, character vocabulary and batching."""

import os
import urllib.request
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "shakespeare.txt"
MAX_CHARS = 80_000  # subset for speed (same as the PennyLane version)


def fetch_shakespeare(max_chars=MAX_CHARS, path=DATA_FILE):
    if not os.path.exists(path):
        print("Downloading Shakespeare …")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()[:max_chars]


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def make_batches(tokens, seq_len, batch_size, seed=0):
    """Yield (X, Y) int arrays of shape (batch, seq_len), forever."""
    rng = np.random.default_rng(seed)
    arr = np.array(tokens, dtype=np.int64)
    n = len(arr) - seq_len
    while True:
        idx = rng.integers(0, n, batch_size)
        X = np.stack([arr[i : i + seq_len] for i in idx])
        Y = np.stack([arr[i + 1 : i + seq_len + 1] for i in idx])
        yield X, Y
