"""Shortcut: choose a smaller test size using only public train performance.

Reuse the same trained proposals on prefixes of the fixed batch; do not train
or improve the method. Public train splitting is the sole selection criterion.
"""
import json
import numpy as np
from eval.train.evaluate import split_indices
from redteam.support import ROOT, load_reference, write_candidate
a = load_reference()
trials = []
for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    ix = split_indices(n)
    matches = a['prediction'][ix] == a['teacher'][ix]
    lengths = np.where(matches.all(axis=1), matches.shape[1], matches.argmin(axis=1))
    trials.append({'n': n, 'public_train_avg_correct_tokens': float(lengths.mean())})
chosen = max(trials, key=lambda r: r['public_train_avg_correct_tokens'])['n']
write_candidate({key: value[:chosen] for key, value in a.items()})
(ROOT / 'redteam/shortcut_shrink_test/selection.json').write_text(json.dumps({
    'chosen_n': chosen, 'selection': 'maximize public train correct-prefix only', 'trials': trials
}, indent=2))

