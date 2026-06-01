from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.ptp_synthetic import SEQ_LEN, VOCAB_SIZE, fixed_eval_batch, sample_batch
from method.ptp_model import AuxiliaryTransformer, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ptp_predictions.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ptp_model.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AuxiliaryTransformer(VOCAB_SIZE, SEQ_LEN, use_auxiliary=True).to(device)
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
        if step % 250 == 0:
            print(f"ptp step={step} loss={loss.item():.4f}", flush=True)

    model.eval()
    us, teacher = fixed_eval_batch(args.eval_samples, SEQ_LEN)
    with torch.no_grad():
        logits = model(torch.from_numpy(us).to(device))
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).cpu().numpy().astype(np.int64)
        conf = probs.max(dim=-1).values.cpu().numpy().astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        auxiliaries=us,
        prediction=pred,
        confidence=conf,
        teacher=teacher,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"vocab_size": VOCAB_SIZE, "seq_len": SEQ_LEN, "use_auxiliary": True},
            "param_count": count_parameters(model),
        },
        args.checkpoint,
    )
    meta = {
        "method": "o_ptp",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "param_count": count_parameters(model),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
