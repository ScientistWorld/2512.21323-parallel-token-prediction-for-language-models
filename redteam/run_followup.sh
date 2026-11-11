#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
trap 'python -m redteam.runner --restore' EXIT
# Preserve the submitted forged score file through the exact hidden-test call.
bash redteam/shortcut_forged_fields/run.sh
# The target-replacement shortcut is also checked without loading reference
# predictions: only the public seed and dimensions construct its inputs.
bash redteam/shortcut_replace_teacher/run.sh
python -m redteam.ranking > redteam/ranks.txt
