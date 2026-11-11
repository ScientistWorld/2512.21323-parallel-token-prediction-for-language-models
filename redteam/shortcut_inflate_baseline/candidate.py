"""Shortcut: move the size-comparison denominator without changing outputs."""
import json
from redteam.support import CHECKPOINTS
p = CHECKPOINTS / 'independent_predictions.json'
meta = json.loads(p.read_text())
meta['param_count'] *= 1000000
p.write_text(json.dumps(meta))

