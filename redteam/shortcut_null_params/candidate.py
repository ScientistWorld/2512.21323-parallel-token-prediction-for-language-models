"""Shortcut attempt: null model-size metadata on unchanged proposals."""
from redteam.support import load_reference, write_candidate
write_candidate(load_reference(), param_count=None)

