#!/bin/bash
# CPU validation job submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Scores existing method prediction artifacts and validates the score schema.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/evaluate_train.sh "$REPO_ROOT/checkpoints"
bash scripts/evaluate_test.sh "$REPO_ROOT/checkpoints"
bash scripts/evaluate.sh
python validate.py --compare
