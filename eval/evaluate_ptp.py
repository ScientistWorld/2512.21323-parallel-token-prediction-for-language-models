from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def prefix_lengths(teacher: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matches = teacher == prediction
    lengths = np.zeros(matches.shape[0], dtype=np.int64)
    for i, row in enumerate(matches):
        wrong = np.flatnonzero(~row)
        lengths[i] = int(wrong[0]) if wrong.size else row.size
    return lengths


def score_file(path: Path) -> dict[str, float]:
    arr = np.load(path)
    teacher = arr["teacher"]
    prediction = arr["prediction"]
    lengths = prefix_lengths(teacher, prediction)
    token_accuracy = float((teacher == prediction).mean())
    confidence = float(arr["confidence"].mean()) if "confidence" in arr else 0.0
    meta_path = path.with_suffix(".json")
    param_count = json.loads(meta_path.read_text()).get("param_count", 0) if meta_path.exists() else 0
    return {
        "avg_correct_tokens": float(lengths.mean()),
        "avg_accepted_tokens": float((lengths + 1).clip(max=teacher.shape[1]).mean()),
        "_token_accuracy": token_accuracy,
        "_mean_confidence": confidence,
        "param_count_M": float(param_count) / 1_000_000.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptp", type=Path, default=Path("checkpoints/ptp_predictions.npz"))
    parser.add_argument("--independent", type=Path, default=Path("checkpoints/independent_predictions.npz"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    if args.ptp.exists():
        results["o_ptp"] = score_file(args.ptp)
    if args.independent.exists():
        results["independent_prediction"] = score_file(args.independent)
    if not results:
        raise FileNotFoundError("No prediction artifacts found to evaluate.")
    if "independent_prediction" in results:
        baseline_params = max(results["independent_prediction"]["param_count_M"], 1e-12)
    else:
        baseline_params = max(next(iter(results.values()))["param_count_M"], 1e-12)
    public_results = {}
    for name, values in results.items():
        public_results[name] = {
            "avg_correct_tokens": values["avg_correct_tokens"],
            "avg_accepted_tokens": values["avg_accepted_tokens"],
            "relative_param_multiplier": values["param_count_M"] / baseline_params,
        }
    scores = {"experiments": {"codecontests_auxiliary_ablation": {"results": public_results}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
