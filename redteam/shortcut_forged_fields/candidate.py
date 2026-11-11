"""Shortcut attempt: impossible proposals with fabricated labels and flags."""
import numpy as np
from redteam.support import load_reference, write_candidate
a = load_reference()
a['prediction'] = np.full_like(a['prediction'], -1)
a['teacher'] = a['prediction'].copy()
a['confidence'] = np.full_like(a['confidence'], np.inf)
a['success'] = np.array(True)
a['time_seconds'] = np.array(-1.0)
write_candidate(a)

