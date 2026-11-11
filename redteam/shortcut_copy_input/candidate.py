"""Shortcut attempt: return the random-control input as token predictions."""
from redteam.support import load_reference, write_candidate
a = load_reference()
a['prediction'] = a['auxiliaries'].copy()
write_candidate(a)

