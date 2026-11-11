"""Train honest variants from the final method code (resumable).

Each variant changes one thing relative to the reference recipe
(seed 13, 1200 steps, batch 192, lr 2e-3, hidden 96 x 3 layers).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "redteam" / "hardening" / "artifacts"

VARIANTS = {
    # name: extra arguments for method.train_ptp
    "weaker_training": ["--steps", "120"],
    "seed14": ["--seed", "14"],
    "seed15": ["--seed", "15"],
    "arch_4layer_h84": ["--config-json", json.dumps({"hidden_size": 84, "layers": 4, "heads": 4})],
    "better_training": ["--steps", "3600"],
}


def main() -> None:
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        out = ARTIFACTS / name / "o_ptp_proposer.pt"
        if out.exists():
            print(f"skip {name}: exists", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        args = [sys.executable, "-m", "method.train_ptp", *VARIANTS[name], "--output", str(out)]
        print("RUN", " ".join(args), flush=True)
        with (out.parent / "train.log").open("w") as log:
            rc = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"done {name}: rc={rc} seconds={time.monotonic() - started:.0f}", flush=True)


if __name__ == "__main__":
    main()
