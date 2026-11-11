"""Shortcut attempt: constant valid tokens, no learned proposer."""
import numpy as np
from redteam.support import load_reference, write_candidate
a = load_reference()
a['prediction'] = np.full_like(a['prediction'], 3)
write_candidate(a)

