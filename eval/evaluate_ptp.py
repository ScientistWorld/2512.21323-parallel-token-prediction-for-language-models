from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.ptp_synthetic import fixed_eval_batch


def prefix_lengths(teacher: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matches = teacher == prediction
    lengths = np.zeros(matches.shape[0], dtype=np.int64)
    for i, row in enumerate(matches):
        wrong = np.flatnonzero(~row)
        lengths[i] = int(wrong[0]) if wrong.size else row.size
    return lengths


def score_file(path: Path, proposal_length: int | None = None) -> dict[str, float]:
    arr = np.load(path)
    if "auxiliaries" not in arr:
        raise KeyError(
            f"{path} must contain an 'auxiliaries' array; eval regenerates teacher targets"
        )
    auxiliaries = arr["auxiliaries"]
    expected_auxiliaries, teacher = fixed_eval_batch(auxiliaries.shape[0], auxiliaries.shape[1])
    if not np.array_equal(auxiliaries, expected_auxiliaries):
        raise ValueError(
            f"{path} was not produced on the fixed held-out auxiliary batch"
        )
    prediction = arr["prediction"]
    confidence_arr = arr["confidence"] if "confidence" in arr else None
    if proposal_length is not None:
        teacher = teacher[:, :proposal_length]
        prediction = prediction[:, :proposal_length]
        if confidence_arr is not None:
            confidence_arr = confidence_arr[:, :proposal_length]
    lengths = prefix_lengths(teacher, prediction)
    token_accuracy = float((teacher == prediction).mean())
    confidence = float(confidence_arr.mean()) if confidence_arr is not None else 0.0
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
    full_results = {}
    if args.ptp.exists():
        full_results["o_ptp"] = score_file(args.ptp)
    if args.independent.exists():
        full_results["independent_prediction"] = score_file(args.independent)
    if not full_results:
        raise FileNotFoundError("No prediction artifacts found to evaluate.")

    if "independent_prediction" in full_results:
        baseline_params = max(full_results["independent_prediction"]["param_count_M"], 1e-12)
    else:
        baseline_params = max(next(iter(full_results.values()))["param_count_M"], 1e-12)

    def public(values_by_method: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        public_results = {}
        for name, values in values_by_method.items():
            public_results[name] = {
                "avg_correct_tokens": values["avg_correct_tokens"],
                "avg_accepted_tokens": values["avg_accepted_tokens"],
                "relative_param_multiplier": values["param_count_M"] / baseline_params,
            }
        return public_results

    experiments = {
        "synthetic_auxiliary_ablation": {"results": public(full_results)},
    }

    for proposal_length in (8, 16):
        truncated = {}
        if args.ptp.exists():
            truncated["o_ptp"] = score_file(args.ptp, proposal_length=proposal_length)
        if args.independent.exists():
            truncated["independent_prediction"] = score_file(
                args.independent, proposal_length=proposal_length
            )
        experiments[f"synthetic_restricted_compute_n{proposal_length}"] = {
            "results": public(truncated)
        }

    scores = {"experiments": experiments}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
