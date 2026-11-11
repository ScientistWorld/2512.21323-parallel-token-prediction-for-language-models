#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
trap 'python -m redteam.runner --restore' EXIT
# Preserve the submitted forged score file through the exact hidden-test call.
bash redteam/shortcut_forged_fields/run.sh
python -m redteam.ranking > redteam/ranks.txt
