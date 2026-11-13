"""Train the reference, the baseline and every audit variant from the final code.

Second hardening iteration (REVIEW.md: training compute was unbudgeted). Every
task is one call of the final training scripts with default arguments except
the ones listed, so each variant changes exactly one thing relative to the
reference recipe (seed 13, 150k steps, batch 512, AdamW 2e-3 peak, 2k warmup,
cosine decay, hidden 96 x 3 layers, ar+lin, argmax).

Tasks run as subprocesses spread over the visible GPUs (several per GPU, the
model is tiny and launch-bound). The run is resumable: a task whose artifact
exists is skipped. No new task starts after ``--start-deadline`` seconds, and
running tasks are stopped after ``--kill-deadline`` seconds, so the audit that
follows always has time to run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "redteam" / "hardening2" / "artifacts"
REFERENCE_STEPS = 150_000


def cfg(**overrides) -> list[str]:
    return ["--config-json", json.dumps(overrides, sort_keys=True)]


# name -> (module, extra args, output path, steps for scheduling, description)
TASKS: dict[str, tuple[str, list[str], Path, int, str]] = {
    "reference": ("method.train_ptp", [], ROOT / "checkpoints" / "o_ptp_proposer.pt",
                  REFERENCE_STEPS, "final reference O-PTP (default arguments)"),
    "baseline": ("baseline.train_independent", [],
                 ROOT / "checkpoints" / "independent_prediction_proposer.pt",
                 REFERENCE_STEPS, "final independent baseline (default arguments)"),
    "train_2x": ("method.train_ptp", ["--steps", "300000"], None, 300_000,
                 "more training alone: same recipe, 2x steps (300k)"),
    "seed14": ("method.train_ptp", ["--seed", "14"], None, REFERENCE_STEPS, "seed 14"),
    "seed15": ("method.train_ptp", ["--seed", "15"], None, REFERENCE_STEPS, "seed 15"),
    "arch_4layer_h84": ("method.train_ptp", cfg(hidden_size=84, layers=4, heads=4), None,
                        REFERENCE_STEPS, "4 layers x 84 wide (the paper's depth), under the cap"),
    "ladder_75k": ("method.train_ptp", ["--steps", "75000"], None, 75_000, "1/2 training"),
    "ladder_37k": ("method.train_ptp", ["--steps", "37500"], None, 37_500, "1/4 training"),
    "ladder_19k": ("method.train_ptp", ["--steps", "18750"], None, 18_750, "1/8 training"),
    "ladder_9k": ("method.train_ptp", ["--steps", "9375"], None, 9_375, "1/16 training"),
    "ar_nn": ("method.train_ptp", cfg(aux_head="nn"), None, REFERENCE_STEPS,
              "[ar+NN] auxiliary embedding (paper Table 7)"),
    "cptp": ("method.train_ptp", cfg(aux_offset=1, decode="pick"), None, REFERENCE_STEPS,
             "C-PTP: position k sees u_1..u_k-1, inverse-CDF decode (paper Eq. 7/11)"),
    "th_lin": ("method.train_ptp", cfg(aux_embedding="th"), None, REFERENCE_STEPS,
               "[th+lin] auxiliary embedding (paper Table 7, weaker than ar+lin)"),
    "train_2x_seed14": ("method.train_ptp", ["--steps", "300000", "--seed", "14"], None, 300_000,
                        "more training alone, second seed"),
}


def output_of(name: str) -> Path:
    out = TASKS[name][2]
    return out if out is not None else ARTIFACTS / name / "o_ptp_proposer.pt"


def log_of(name: str) -> Path:
    return ARTIFACTS / name / "train.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--start-deadline", type=float, default=13_000.0)
    parser.add_argument("--kill-deadline", type=float, default=16_500.0)
    parser.add_argument("--per-gpu", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    n_gpu = torch.cuda.device_count()
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    workers = max(1, min(n_gpu * args.per_gpu, n_cpu)) if n_gpu else 1
    names = args.names or list(TASKS)
    pending = [n for n in names if not output_of(n).exists()]
    print(f"gpus={n_gpu} cpus={n_cpu} workers={workers} pending={pending}", flush=True)

    started = time.monotonic()
    running: dict[str, tuple[subprocess.Popen, float, object]] = {}
    slot = 0
    summary_path = ARTIFACTS / "train_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    while pending or running:
        elapsed = time.monotonic() - started
        while pending and len(running) < workers and elapsed < args.start_deadline:
            name = pending.pop(0)
            module, extra, _, _, _ = TASKS[name]
            device = f"cuda:{slot % n_gpu}" if n_gpu else "cpu"
            slot += 1
            out = output_of(name)
            out.parent.mkdir(parents=True, exist_ok=True)
            log_of(name).parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", module, *extra, "--device", device]
            if TASKS[name][2] is None:
                cmd += ["--output", str(out)]
            handle = log_of(name).open("w")
            handle.write("RUN " + " ".join(cmd) + "\n")
            handle.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
            running[name] = (proc, time.monotonic(), handle)
            print(f"[{elapsed:7.0f}s] start {name} on {device}", flush=True)
        if elapsed >= args.start_deadline and pending:
            print(f"[{elapsed:7.0f}s] start deadline: not running {pending}", flush=True)
            for name in pending:
                summary[name] = {"status": "not_started"}
            pending = []
        for name in list(running):
            proc, t0, handle = running[name]
            if proc.poll() is None and elapsed >= args.kill_deadline:
                proc.kill()
                proc.wait()
            if proc.poll() is not None:
                handle.close()
                rc = proc.returncode
                summary[name] = {"rc": rc, "seconds": round(time.monotonic() - t0, 1),
                                 "artifact": output_of(name).exists(),
                                 "description": TASKS[name][4]}
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, indent=2))
                print(f"[{elapsed:7.0f}s] done {name}: {summary[name]}", flush=True)
                del running[name]
        time.sleep(5)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
