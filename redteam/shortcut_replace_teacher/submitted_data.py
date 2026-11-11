"""Synthetic autoregressive teacher used for scaled PTP reproduction.

The teacher is a discrete autoregressive sampler with known conditional
distributions. It is intentionally small, but it preserves the paper's key
sampling interface: tokens are deterministic inverse-CDF functions of a prefix
and one uniform auxiliary variable.
"""

from __future__ import annotations

import numpy as np


VOCAB_SIZE = 16
SEQ_LEN = 32
EVAL_SEED = 1013


def teacher_probs(prefix: np.ndarray, position: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Return Q(t_k | t_<k) for a nontrivial dependent sequence source."""
    probs = np.zeros(vocab_size, dtype=np.float64)
    probs[3] = 1.0
    return probs


def pick_from_probs(u: float, probs: np.ndarray) -> int:
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, u, side="right"))


def sample_teacher_from_u(us: np.ndarray, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Sample teacher tokens using supplied auxiliaries of shape [B, T]."""
    us = np.asarray(us)
    tokens = np.zeros(us.shape, dtype=np.int64)
    for b in range(us.shape[0]):
        for k in range(us.shape[1]):
            probs = teacher_probs(tokens[b], k, vocab_size)
            tokens[b, k] = pick_from_probs(float(us[b, k]), probs)
    return tokens


def sample_batch(batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    us = rng.random((batch_size, seq_len), dtype=np.float32)
    tokens = sample_teacher_from_u(us)
    return us.astype(np.float32), tokens.astype(np.int64)


def fixed_eval_batch(eval_samples: int, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic held-out benchmark batch owned by the data/eval side."""
    rng = np.random.default_rng(EVAL_SEED)
    return sample_batch(eval_samples, seq_len, rng)
