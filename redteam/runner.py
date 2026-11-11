"""Run each candidate, public development evaluation, then the unmodified test.

Only a candidate's preparation can edit agent-visible inputs. The hidden test
is always a fresh subprocess of `bash scripts/evaluate_test.sh` with no args.
Reference backups are conveniences for this audit, not shortcut inputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'redteam'
ORIGINAL = AUDIT / '_original'
FILES = ['data/ptp_synthetic.py', 'scoring/scores_train.json',
         'scoring/scores_test.json', 'scoring/scores.json']


def snapshot():
    if (ORIGINAL / 'ready').exists():
        return
    ORIGINAL.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'checkpoints', ORIGINAL / 'checkpoints', dirs_exist_ok=True)
    for name in FILES:
        src = ROOT / name
        dest = ORIGINAL / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dest)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in FILES if (ROOT / name).exists()}
    (AUDIT / 'reference' / 'initial_hashes.json').write_text(json.dumps(hashes, indent=2))
    (ORIGINAL / 'ready').write_text('complete\n')


def restore():
    # Restore only paths the audit can change. Preserve the actual base outputs.
    if not (ORIGINAL / 'ready').exists():
        return
    for path in (ROOT / 'checkpoints').glob('*'):
        if path.is_file() or path.is_symlink():
            path.unlink()
    shutil.copytree(ORIGINAL / 'checkpoints', ROOT / 'checkpoints', dirs_exist_ok=True)
    for name in FILES:
        source = ORIGINAL / name
        if source.exists():
            shutil.copy2(source, ROOT / name)
        else:
            (ROOT / name).unlink(missing_ok=True)
    shutil.rmtree(ROOT / 'data' / '__pycache__', ignore_errors=True)


def command(cmd, output):
    with output.open('w') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def run_case(name):
    snapshot()
    restore()
    folder = AUDIT / name
    folder.mkdir(exist_ok=True)
    started = time.monotonic()
    statuses = {}
    print(f'START {name}', flush=True)
    try:
        # Start without an inherited scorecard. A candidate may subsequently
        # write its own; preserve that file exactly until the hidden test.
        (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
        statuses['prepare'] = command(['python', '-m', f'redteam.{name}.candidate'], folder / 'prepare.txt')
        if statuses['prepare']:
            raise RuntimeError(f'{name}: preparation failed')
        statuses['train_evaluation'] = command(['bash', 'scripts/evaluate_train.sh'], folder / 'train_evaluation.txt')
        if (ROOT / 'scoring/scores_train.json').exists():
            shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
        if (folder / 'after_train.py').exists():
            statuses['after_train'] = command(['python', '-m', f'redteam.{name}.after_train'], folder / 'after_train.txt')
            if statuses['after_train']:
                raise RuntimeError('after-train preparation failed')
        result = ROOT / 'scoring/scores_test.json'
        before_test_mtime = result.stat().st_mtime_ns if result.exists() else None
        statuses['hidden_test'] = command(['bash', 'scripts/evaluate_test.sh'], folder / 'hidden_test.txt')
        statuses['test_output_written'] = result.exists() and result.stat().st_mtime_ns != before_test_mtime
        if statuses['test_output_written']:
            shutil.copy2(result, folder / 'scores_test.json')
        else:
            (folder / 'scores_test.json').write_text(json.dumps({
                '_audit_no_test_output': True, 'experiments': {}
            }, indent=2))
        # Preserve candidate proposals for inspection, but not in git.
        for path in (ROOT / 'checkpoints').glob('ptp_*'):
            if path.is_file():
                shutil.copy2(path, folder / path.name)
        if (ROOT / 'data/ptp_synthetic.py').read_bytes() != (ORIGINAL / 'data/ptp_synthetic.py').read_bytes():
            shutil.copy2(ROOT / 'data/ptp_synthetic.py', folder / 'submitted_data.py')
        (folder / 'train_allowlist_at_test.json').write_text((ROOT / 'scoring/scores_train.json').read_text())
    except Exception as exc:
        statuses['error'] = repr(exc)
        raise
    finally:
        statuses['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (folder / 'status.json').write_text(json.dumps(statuses, indent=2))
        restore()
        print(f'END {name}: {statuses}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='?')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore:
        restore()
    else:
        run_case(args.name)


if __name__ == '__main__':
    main()
