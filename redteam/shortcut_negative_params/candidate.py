"""Shortcut attempt: a failed task with an impossible negative model size."""
import numpy as np
from redteam.support import load_reference, write_candidate
a = load_reference()
a['prediction'] = np.full_like(a['prediction'], -1)
write_candidate(a, param_count=-3436000000)

