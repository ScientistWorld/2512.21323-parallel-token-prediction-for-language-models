"""Distil a One-Hot PTP proposer from the autoregressive teacher (Alg. 3).

Each step samples auxiliaries u ~ U[0,1]^T, lets the teacher pick its tokens by
inverse-CDF sampling with those auxiliaries (Eq. 2/3; teacher samples are the
paper's lowest-variance training source, App. E.2), and minimises the O-PTP
cross-entropy of Eq. 10 for every future position in one causal forward pass.
No context is given, as in the paper's App. G.2 setting. The trained proposer is
written as a contract artifact for the evaluators.

Training length follows the paper's App. G.2 (150k steps). With fresh teacher
samples every step, the batch (512) and the warmup + cosine learning-rate decay
are chosen so that the proposer is trained to convergence: training the same
recipe twice as long does not improve it beyond seed-to-seed variation. A GPU is
used when available (about 15 minutes); the CPU fallback is much slower.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from data.ptp_synthetic import VOCAB_SIZE  # noqa: E402
from data.teacher_stream import TeacherStream  # noqa: E402
from method.ptp_model import build_o_ptp, count_parameters, save_proposer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    parser.add_argument("--min-lr", type=float, default=0.0, help="final learning rate")
    parser.add_argument("--warmup", type=int, default=2_000, help="linear warmup steps")
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--config-json", type=str, default="",
                        help="optional JSON overrides of the O-PTP config")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/o_ptp_proposer.pt"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if args.warmup > 0 and step <= args.warmup:
        return args.lr * step / args.warmup
    if args.schedule == "constant":
        return args.lr
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_determinism(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # The math attention kernel is deterministic; sequences are only 32 long.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    setup_determinism(device)
    torch.manual_seed(args.seed)

    stream = TeacherStream(args.batch_size, args.seed, device)
    stream.check()

    overrides = json.loads(args.config_json) if args.config_json else {}
    model = build_o_ptp(overrides).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    started = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        us_t, y_t = stream.next()
        logits = model(us_t)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y_t.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            print(f"ptp step={step} lr={lr:.2e} loss={loss.item():.4f} "
                  f"seconds={time.monotonic() - started:.0f}", flush=True)

    model.eval()
    model.cpu()
    save_proposer(args.output, model)
    meta = {
        "method": "o_ptp",
        "config": model.config,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup": args.warmup,
        "schedule": args.schedule,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "train_seconds": round(time.monotonic() - started, 1),
        "param_count": count_parameters(model),
        "note": "provenance only; evaluators never read this file",
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
