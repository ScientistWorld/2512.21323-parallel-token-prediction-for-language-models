#!/bin/bash
# Hidden test. Takes no arguments and scores every proposer artifact on disk
# (checkpoints/*_proposer.pt) on the control batch held by eval/test/, against
# eval/test/'s own frozen teacher. Writes scoring/scores_test.json.
#
# Python runs isolated (-I): neither the working directory, PYTHONPATH nor the
# user site-packages can shadow the evaluator's imports. A stale scorecard is
# removed first; if the evaluator itself cannot run, the canonical slots are
# written with worst values so a failure never scores as high or as missing.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p scoring
rm -f scoring/scores_test.json scoring/scores_test.json.tmp

python -I "$REPO_ROOT/eval/test/evaluate.py"
status=$?

if [ "$status" -ne 0 ] || [ ! -s scoring/scores_test.json ]; then
    echo "evaluate_test: evaluator failed (status $status); writing worst values" >&2
    worst='{"avg_accepted_tokens": 1.0, "avg_correct_tokens": 0.0, "token_accuracy": 0.0, "tail_token_accuracy": 0.0, "prefix8_success_rate": 0.0}'
    cat > scoring/scores_test.json <<JSON
{
  "slice": "test",
  "experiments": {
    "synthetic_auxiliary_ablation": {"results": {"independent_prediction": $worst, "o_ptp": $worst}},
    "synthetic_restricted_compute_n8": {"results": {"independent_prediction": $worst, "o_ptp": $worst}}
  },
  "_evaluator_failed": true
}
JSON
    exit 1
fi
