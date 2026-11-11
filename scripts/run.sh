#!/bin/bash
# Reference job (runs in the gym container, workspace at /home/user, no
# internet): trains the independent baseline and O-PTP from scratch (each
# writes checkpoints/<name>_proposer.pt and runs scripts/evaluate.sh), then
# writes the train- and test-slice scorecards and validates the workspace.
# The hardening audit that checked this gym is redteam/hardening/run_job.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# One allocated CPU: avoid BLAS/OpenMP thread oversubscription.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --post-design
