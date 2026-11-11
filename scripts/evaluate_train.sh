#!/bin/bash
# Train-slice (development) evaluator. Same protocol and code path as the
# hidden test, on the public control batch. Scores every proposer artifact
# under checkpoints/ (or --checkpoints-dir DIR) and writes
# scoring/scores_train.json (or --output FILE).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p scoring

python -I "$REPO_ROOT/eval/train/evaluate.py" "$@"
