#!/bin/bash
# One GPU job for the second hardening iteration (REVIEW.md: training compute
# was unbudgeted, so more training alone was the largest gain).
#   1. train the reference and the baseline from the final code (default
#      arguments: 150k steps to convergence) plus every audit variant, spread
#      over the visible GPUs;
#   2. write scoring/scores{,_train,_test}.json from the reference runs alone;
#   3. run every variant and the shortcut suite through the unmodified
#      no-argument hidden test, paired comparisons, seed spread, ladder, ranks.
# Every stage is resumable and saves its outputs as it goes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

stamp "environment"
nproc
python - <<'PY'
import os, torch, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count(), "numpy", numpy.__version__)
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i))
try:
    print("affinity cpus", len(os.sched_getaffinity(0)))
except Exception as exc:
    print("affinity unknown", exc)
PY
if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"; then
    stamp "no CUDA device visible: stopping (the converged reference needs a GPU)"
    exit 3
fi

# Keep the previous 1,200-step reference and baseline as audit variants.
OLD=redteam/hardening2/artifacts/old_reference
if [ ! -d "$OLD" ]; then
    mkdir -p "$OLD"
    for f in o_ptp_proposer.pt o_ptp_proposer.json \
             independent_prediction_proposer.pt independent_prediction_proposer.json; do
        if [ -f "checkpoints/$f" ]; then mv "checkpoints/$f" "$OLD/$f"; fi
    done
fi

stamp "stage 1: reference, baseline and variants (GPU pool)"
python -m redteam.hardening2.train_variants

stamp "stage 2: scorecards from the reference runs"
if [ -f checkpoints/o_ptp_proposer.pt ] && [ -f checkpoints/independent_prediction_proposer.pt ]; then
    bash scripts/evaluate.sh || stamp "dev evaluation (scores.json) failed"
    bash scripts/evaluate_train.sh || stamp "train evaluation failed"
    bash scripts/evaluate_test.sh || stamp "test evaluation failed"
    cat scoring/scores_test.json
else
    stamp "reference or baseline missing: scorecards not regenerated"
    exit 4
fi

stamp "stage 3: audit through the hidden test"
python -m redteam.hardening2.audit run
python -m redteam.hardening2.audit paired
python -m redteam.hardening2.audit rank

stamp "validate"
python validate.py --post-design || true
python validate.py || true
stamp "done"
