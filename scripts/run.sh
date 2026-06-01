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

cd /home/user

bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
