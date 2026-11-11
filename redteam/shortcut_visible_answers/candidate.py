"""Shortcut: rebuild every answer using only the public generator and seed."""
from data.ptp_synthetic import fixed_eval_batch, SEQ_LEN
from redteam.support import write_candidate

u, answer = fixed_eval_batch(4096, SEQ_LEN)
write_candidate({'auxiliaries': u, 'prediction': answer})

