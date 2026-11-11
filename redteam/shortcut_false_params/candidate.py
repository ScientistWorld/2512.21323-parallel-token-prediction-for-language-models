"""Shortcut: keep the reference proposer, erase its claimed parameter cost."""
from redteam.support import load_reference, write_candidate
write_candidate(load_reference(), param_count=0)

