#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

python -m baseline.train_independent --steps 1200 --batch-size 192 --eval-samples 4096
python -m method.train_ptp --steps 1200 --batch-size 192 --eval-samples 4096
bash scripts/evaluate.sh
