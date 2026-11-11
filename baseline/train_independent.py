"""Independent multi-token prediction baseline (paper Sec. 4.2.2, App. G.2).

Identical training code and network shape as O-PTP, but every position gets an
uninformative learned input in place of its auxiliary ("[MASK]"), so later
tokens cannot be coordinated with the sampled earlier ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.proposer import ProposerNet, count_parameters, save_proposer
from data.ptp_synthetic import SEQ_LEN, VOCAB_SIZE, sample_batch

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
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/independent_prediction_proposer.pt")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProposerNet(INDEPENDENT_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for step in range(1, args.steps + 1):
        us, tokens = sample_batch(args.batch_size, SEQ_LEN, rng)
        us_t = torch.from_numpy(us).to(device)
        y_t = torch.from_numpy(tokens).to(device)
        logits = model(us_t)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y_t.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            print(f"independent step={step} loss={loss.item():.4f}", flush=True)

    model.eval()
    model.cpu()
    save_proposer(args.output, model)
    meta = {
        "method": "independent_prediction",
        "config": model.config,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "param_count": count_parameters(model),
        "note": "provenance only; evaluators never read this file",
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
