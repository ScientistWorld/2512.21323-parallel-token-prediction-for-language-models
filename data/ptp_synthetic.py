"""Synthetic autoregressive teacher used for scaled PTP reproduction.

The teacher is a discrete autoregressive sampler with known conditional
distributions. It is intentionally small, but it preserves the paper's key
sampling interface: tokens are deterministic inverse-CDF functions of a prefix
and one uniform auxiliary variable (Eq. 2/3 of the paper).

This module is the public copy used for training and development. The scored
evaluators keep their own frozen copies of the same teacher, so editing this
file changes only what a method trains on, never what it is scored against.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


VOCAB_SIZE = 16
SEQ_LEN = 32
# Public development batch seed (the scored test batch is held by eval/test/).
DEV_SEED = 1013
EVAL_SEED = DEV_SEED  # backward-compatible alias


def teacher_probs(prefix: np.ndarray, position: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Return Q(t_k | t_<k) for a nontrivial dependent sequence source."""
    probs = np.full(vocab_size, 0.02 / (vocab_size - 2), dtype=np.float64)
    if position == 0:
        main = 3
        alt = 11
    else:
        prev = int(prefix[position - 1])
        prev2 = int(prefix[position - 2]) if position >= 2 else 5
        if position % 5 == 0:
            main = (prev + prev2 + position) % vocab_size
        elif position % 3 == 0:
            main = (2 * prev + position) % vocab_size
        else:
            main = (prev + 4 + (position % 4)) % vocab_size
        alt = (main + 7 + prev2) % vocab_size
        if alt == main:
            alt = (main + 1) % vocab_size
    probs[main] = 0.78
    probs[alt] = 0.20
    probs /= probs.sum()
    return probs


def pick_from_probs(u: float, probs: np.ndarray) -> int:
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, u, side="right"))


@lru_cache(maxsize=4)
def cdf_table(seq_len: int = SEQ_LEN, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """CDF of Q(. | prev, prev2) for every position: shape [T, V, V, V].

    Built from ``teacher_probs`` itself, so table lookups reproduce the scalar
    definition exactly (same float64 operations per state).
    """
    table = np.empty((seq_len, vocab_size, vocab_size, vocab_size), dtype=np.float64)
    prefix = np.zeros(max(seq_len, 2), dtype=np.int64)
    for k in range(seq_len):
        for prev in range(vocab_size):
            for prev2 in range(vocab_size):
                prefix[:] = 0
                if k >= 1:
                    prefix[k - 1] = prev
                if k >= 2:
                    prefix[k - 2] = prev2
                table[k, prev, prev2] = np.cumsum(teacher_probs(prefix, k, vocab_size))
    table.setflags(write=False)
    return table


def sample_teacher_from_u(us: np.ndarray, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Sample teacher tokens using supplied auxiliaries of shape [B, T].

    Vectorised over the batch; identical to applying ``pick_from_probs`` with
    ``teacher_probs`` position by position.
    """
    us = np.asarray(us)
    batch, seq_len = us.shape
    table = cdf_table(seq_len, vocab_size)
    u64 = us.astype(np.float64)
    tokens = np.zeros((batch, seq_len), dtype=np.int64)
    zeros = np.zeros(batch, dtype=np.int64)
    for k in range(seq_len):
        prev = tokens[:, k - 1] if k >= 1 else zeros
        prev2 = tokens[:, k - 2] if k >= 2 else zeros
        cdf = table[k, prev, prev2]
        tokens[:, k] = np.minimum((cdf <= u64[:, k : k + 1]).sum(axis=1), vocab_size - 1)
    return tokens


def sample_batch(batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    us = rng.random((batch_size, seq_len), dtype=np.float32)
    tokens = sample_teacher_from_u(us)
    return us.astype(np.float32), tokens.astype(np.int64)


def fixed_eval_batch(eval_samples: int, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic public development batch (same one eval/train/ scores on)."""
    rng = np.random.default_rng(DEV_SEED)
    return sample_batch(eval_samples, seq_len, rng)
