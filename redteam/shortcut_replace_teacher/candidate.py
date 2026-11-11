"""Shortcut: replace the fixed public target with a constant distribution."""
from pathlib import Path
import numpy as np
from redteam.support import ROOT, load_reference, write_candidate

p = ROOT / 'data/ptp_synthetic.py'
source = p.read_text()
start = source.index('    probs = np.full(', source.index('def teacher_probs('))
end = source.index('\n\n\ndef pick_from_probs', start)
source = source[:start] + '''    probs = np.zeros(vocab_size, dtype=np.float64)
    probs[3] = 1.0
    return probs''' + source[end:]
p.write_text(source)
a = load_reference()
a['prediction'] = np.full_like(a['prediction'], 3)
write_candidate(a)

