#!/bin/bash
# One CPU job for the hardening iteration:
#   1. retrain reference O-PTP and the independent baseline from the final code
#      and write scoring/scores_{train,test}.json from those two runs only;
#   2. run every available honest variant and shortcut through the unmodified
#      no-argument hidden test (isolated workspace copies);
#   3. train honest variants from the final code; 4. score them, paired stats,
#      ranks. Each stage is resumable and saves its outputs as it goes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p checkpoints/legacy
for f in ptp_model.pt independent_model.pt ptp_predictions.npz ptp_predictions.json \
         independent_predictions.npz independent_predictions.json; do
    if [ -f "checkpoints/$f" ]; then mv "checkpoints/$f" "checkpoints/legacy/$f"; fi
done

if [ ! -f checkpoints/.hardening_stage1_done ]; then
    stamp "stage 1: reference + baseline from final code"
    bash scripts/reproduce.sh || { stamp "reproduce failed"; exit 1; }
    touch checkpoints/.hardening_stage1_done
fi
bash scripts/evaluate_train.sh || stamp "train evaluation failed"
bash scripts/evaluate_test.sh || stamp "test evaluation failed"
cat scoring/scores_test.json

stamp "stage 2: audit with available artifacts"
python -m redteam.hardening.audit run
python -m redteam.hardening.audit rank

stamp "stage 3: honest variants from final code"
python -m redteam.hardening.train_variants

stamp "stage 4: audit new variants, paired comparisons, ranks"
python -m redteam.hardening.audit run
python -m redteam.hardening.audit paired
python -m redteam.hardening.audit rank

stamp "validate"
python validate.py --post-design || true
stamp "done"
