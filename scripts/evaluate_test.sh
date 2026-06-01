#!/bin/bash
# Held-out test-slice evaluator.
#
# It scores the same method entries recorded by scoring/scores_train.json when
# that file exists, then writes the held-out report to scoring/scores_test.json.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p scoring

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    predictions_dir="$1"
    shift
    python -m eval.test.evaluate --predictions-dir "$predictions_dir" "$@"
else
    python -m eval.test.evaluate "$@"
fi
