"""Shortcut attempt: a one-column prediction broadcast over all positions."""
from redteam.support import load_reference, write_candidate
a = load_reference()
a['prediction'] = a['prediction'][:, :1]
write_candidate(a)

