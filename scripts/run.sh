#!/bin/bash
# CPU reproduction job submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Regenerates baseline and O-PTP artifacts, scores all public evaluation files,
# and validates the score schema.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh "$REPO_ROOT/checkpoints"
bash scripts/evaluate_test.sh "$REPO_ROOT/checkpoints"
python validate.py --compare
