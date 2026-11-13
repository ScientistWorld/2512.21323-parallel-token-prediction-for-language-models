#!/bin/bash
# Current job: the second hardening audit (REVIEW.md: training compute). It
# retrains the reference and the baseline from the final code (default
# arguments, trained to convergence on a GPU), writes the train/test
# scorecards from those runs, and audits variants and shortcuts through the
# no-argument hidden test. See redteam/hardening2/run_job.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bash redteam/hardening2/run_job.sh
