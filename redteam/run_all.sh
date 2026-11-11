#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# A single CPU is assigned to this job. These affect thread count only.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
trap 'python -m redteam.runner --restore' EXIT
python - <<'PY'
import json
import subprocess
from pathlib import Path
for name in json.loads(Path('redteam/cases.json').read_text()):
    status = Path('redteam') / name / 'status.json'
    if status.exists():
        print('Already saved:', name, flush=True)
        continue
    subprocess.run(['bash', f'redteam/{name}/run.sh'], check=True)
PY
python -m redteam.ranking > redteam/ranks.txt

