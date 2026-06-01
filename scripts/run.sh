#!/bin/bash
# Train/test scoring job submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Scores existing method prediction artifacts without retraining.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/evaluate_train.sh "$REPO_ROOT/checkpoints"
bash scripts/evaluate_test.sh "$REPO_ROOT/checkpoints"
