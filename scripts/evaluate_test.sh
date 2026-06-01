#!/bin/bash
# Held-out test-slice evaluator.
#
# It scores the same method entries recorded by scoring/scores_train.json when
# that file exists, then writes the held-out report to scoring/scores_test.json.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate "$@"
