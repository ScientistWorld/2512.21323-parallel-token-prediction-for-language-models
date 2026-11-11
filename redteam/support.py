"""Audit preparation utilities. No hidden-test imports or evaluator changes."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / 'checkpoints'


def load_reference():
    with np.load(CHECKPOINTS / 'ptp_predictions.npz') as a:
        return {k: a[k].copy() for k in a.files}


def write_candidate(arrays, param_count=343600):
    np.savez_compressed(CHECKPOINTS / 'ptp_predictions.npz', **arrays)
    (CHECKPOINTS / 'ptp_predictions.json').write_text(json.dumps({'param_count': param_count}))


def train(steps, seed):
    subprocess.run([
        'python', '-m', 'method.train_ptp', '--steps', str(steps),
        '--batch-size', '192', '--seed', str(seed),
    ], cwd=ROOT, check=True)


def no_auxiliary():
    # This is the trained, same-size key-idea-removed ablation, not random output.
    shutil.copy2(CHECKPOINTS / 'independent_predictions.npz', CHECKPOINTS / 'ptp_predictions.npz')
    shutil.copy2(CHECKPOINTS / 'independent_predictions.json', CHECKPOINTS / 'ptp_predictions.json')

