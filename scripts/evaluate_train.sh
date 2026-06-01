#!/bin/bash
# Train-slice evaluator for future method development.
#
# Mirrors scripts/evaluate.sh: it reads prediction artifacts from checkpoints/
# (or explicit --ptp/--independent paths) and writes a scoring JSON file.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    predictions_dir="$1"
    shift
    python -m eval.train.evaluate --predictions-dir "$predictions_dir" "$@"
else
    python -m eval.train.evaluate "$@"
fi
