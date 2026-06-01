#!/bin/bash
# Train-slice evaluator for future method development.
#
# Mirrors scripts/evaluate.sh: it reads prediction artifacts from checkpoints/
# (or explicit --ptp/--independent paths) and writes a scoring JSON file.

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate "$@"
