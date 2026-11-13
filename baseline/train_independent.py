"""Independent multi-token prediction baseline (paper Sec. 4.2.2, App. G.2).

Identical training recipe and network shape as O-PTP, but every position gets an
uninformative learned input in place of its auxiliary ("[MASK]"), so later
tokens cannot be coordinated with the sampled earlier ones. Like the O-PTP
reference it is trained to convergence (150k steps of 512 fresh teacher
samples, warmup + cosine learning-rate decay).
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

from data.proposer import ProposerNet, count_parameters, save_proposer  # noqa: E402
from data.ptp_synthetic import VOCAB_SIZE  # noqa: E402
from data.teacher_stream import TeacherStream  # noqa: E402

INDEPENDENT_CONFIG = {
    "hidden_size": 96,
    "layers": 3,
    "heads": 4,
    "ff_mult": 4,
    "aux_embedding": "none",
    "aux_head": "lin",
    "aux_offset": 0,
    "decode": "argmax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/independent_prediction_proposer.pt")
    )
    return parser.parse_args()


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if args.warmup > 0 and step <= args.warmup:
        return args.lr * step / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.manual_seed(args.seed)

    stream = TeacherStream(args.batch_size, args.seed, device)
    stream.check()

    model = ProposerNet(INDEPENDENT_CONFIG).to(device)
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
            print(f"independent step={step} lr={lr:.2e} loss={loss.item():.4f} "
                  f"seconds={time.monotonic() - started:.0f}", flush=True)

    model.eval()
    model.cpu()
    save_proposer(args.output, model)
    meta = {
        "method": "independent_prediction",
        "config": model.config,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup": args.warmup,
        "schedule": "cosine",
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
