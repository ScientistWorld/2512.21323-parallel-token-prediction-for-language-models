"""Shortcut: create a failing proposer, then omit its name from train scores."""
import numpy as np
from redteam.support import load_reference, write_candidate
a = load_reference()
a['prediction'] = np.full_like(a['prediction'], -1)
write_candidate(a)

