"""Frozen copy of the reference teacher owned by the train-slice evaluator.

Identical to the public generator in data/ptp_synthetic.py at the time the gym
was fixed. The train evaluator never imports data/, so editing the public copy
cannot change the targets a proposer is scored against.
"""

from __future__ import annotations

import numpy as np


VOCAB_SIZE = 16
SEQ_LEN = 32


def teacher_probs(prefix: np.ndarray, position: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
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


def cdf_table(seq_len: int = SEQ_LEN, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
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
    return table


def sample_teacher_from_u(us: np.ndarray, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
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
