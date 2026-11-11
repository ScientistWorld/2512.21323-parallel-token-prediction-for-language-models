#!/bin/bash
# Job entry point (runs in the gym container, workspace at /home/user, no
# internet). This job: hardening iteration — retrain the reference and the
# baseline from the final code, regenerate the scorecards, then run the
# hardening audit (honest variants + shortcuts through the hidden test).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
bash redteam/hardening/run_job.sh
