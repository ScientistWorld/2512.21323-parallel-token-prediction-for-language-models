"""Shortcut: replace the fixed public target with a constant distribution."""
from pathlib import Path
import numpy as np
from data.ptp_synthetic import EVAL_SEED, SEQ_LEN
from redteam.support import ROOT, write_candidate

p = ROOT / 'data/ptp_synthetic.py'
source = p.read_text()
start = source.index('    probs = np.full(', source.index('def teacher_probs('))
end = source.index('\n\n\ndef pick_from_probs', start)
source = source[:start] + '''    probs = np.zeros(vocab_size, dtype=np.float64)
    probs[3] = 1.0
    return probs''' + source[end:]
p.write_text(source)
u = np.random.default_rng(EVAL_SEED).random((4096, SEQ_LEN), dtype=np.float32)
write_candidate({'auxiliaries': u, 'prediction': np.full(u.shape, 3, dtype=np.int64)})
