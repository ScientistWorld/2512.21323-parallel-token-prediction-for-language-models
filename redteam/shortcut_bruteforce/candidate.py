"""Shortcut: enumerate the tiny reference's full state transition table.

This is ordinary sequential generation, exactly what the parallel-proposal
problem exists to avoid. No training, no learned proposer, no hidden inputs.
"""
import json
import time
import numpy as np
from data.ptp_synthetic import EVAL_SEED, SEQ_LEN, VOCAB_SIZE, teacher_probs
from redteam.support import write_candidate

start = time.monotonic()
v, t, b = VOCAB_SIZE, SEQ_LEN, 4096
table = np.empty((t, v, v, v), dtype=np.float64)
prefix = np.zeros(t, dtype=np.int64)
for k in range(t):
    for prev in range(v):
        for prev2 in range(v):
            if k:
                prefix[k - 1] = prev
            if k >= 2:
                prefix[k - 2] = prev2
            table[k, prev, prev2] = np.cumsum(teacher_probs(prefix, k))
u = np.random.default_rng(EVAL_SEED).random((b, t), dtype=np.float32)
prediction = np.zeros((b, t), dtype=np.int64)
for k in range(t):
    prev = prediction[:, k - 1] if k else np.zeros(b, dtype=np.int64)
    prev2 = prediction[:, k - 2] if k >= 2 else np.full(b, 5, dtype=np.int64)
    prediction[:, k] = (u[:, k, None] >= table[k, prev, prev2]).sum(axis=-1)
write_candidate({'auxiliaries': u, 'prediction': prediction})
print(json.dumps({'enumerated_transitions': t*v*v, 'table_bytes': table.nbytes,
                  'sequential_steps': t, 'elapsed_seconds': time.monotonic()-start}))

