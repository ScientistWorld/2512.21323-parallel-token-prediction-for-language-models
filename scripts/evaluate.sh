#!/bin/bash
# Standard evaluation entry point. Scores every proposer artifact under
# checkpoints/ (files named <method>_proposer.pt, see briefing/evaluation.md)
# with the public development evaluator in eval/train/ and writes
# scoring/scores.json. Uses only code in eval/; never imports method/.
#
# Usage:
#   bash scripts/evaluate.sh                 # score checkpoints/*_proposer.pt
#   bash scripts/evaluate.sh --checkpoints-dir some/dir

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p scoring

python -I "$REPO_ROOT/eval/train/evaluate.py" --output "$REPO_ROOT/scoring/scores.json" "$@"
