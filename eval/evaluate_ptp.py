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


def token_accuracy(teacher: np.ndarray, prediction: np.ndarray) -> float:
    return float((teacher == prediction).mean())


def tail_token_accuracy(teacher: np.ndarray, prediction: np.ndarray) -> float:
    midpoint = max(teacher.shape[1] // 2, 1)
    return float((teacher[:, midpoint:] == prediction[:, midpoint:]).mean())


def prefix8_success_rate(teacher: np.ndarray, prediction: np.ndarray) -> float:
    horizon = min(8, teacher.shape[1])
    return float(np.all(teacher[:, :horizon] == prediction[:, :horizon], axis=1).mean())


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
            f"{path} was not produced on the fixed held-out random-control batch"
        )
    prediction = arr["prediction"]
    confidence_arr = arr["confidence"] if "confidence" in arr else None
    if proposal_length is not None:
        teacher = teacher[:, :proposal_length]
        prediction = prediction[:, :proposal_length]
        if confidence_arr is not None:
            confidence_arr = confidence_arr[:, :proposal_length]
    lengths = prefix_lengths(teacher, prediction)
    confidence = float(confidence_arr.mean()) if confidence_arr is not None else 0.0
    meta_path = path.with_suffix(".json")
    param_count = json.loads(meta_path.read_text()).get("param_count", 0) if meta_path.exists() else 0
    return {
        "avg_correct_tokens": float(lengths.mean()),
        "avg_accepted_tokens": float((lengths + 1).clip(max=teacher.shape[1]).mean()),
        "token_accuracy": token_accuracy(teacher, prediction),
        "tail_token_accuracy": tail_token_accuracy(teacher, prediction),
        "prefix8_success_rate": prefix8_success_rate(teacher, prediction),
        "_mean_confidence": confidence,
        "param_count_M": float(param_count) / 1_000_000.0,
    }


def artifact_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    aliases = {
        "ptp": "o_ptp",
        "independent": "independent_prediction",
    }
    return aliases.get(stem, stem)


def discover_artifacts(args: argparse.Namespace) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    seen_paths: set[Path] = set()

    explicit = {
        "o_ptp": args.ptp,
        "independent_prediction": args.independent,
    }
    for name, path in explicit.items():
        if path.exists():
            resolved = path.resolve()
            artifacts[name] = path
            seen_paths.add(resolved)

    if args.predictions_dir.exists():
        for path in sorted(args.predictions_dir.glob("*_predictions.npz")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            artifacts.setdefault(artifact_name(path), path)
            seen_paths.add(resolved)

    if not artifacts:
        raise FileNotFoundError(
            f"No prediction artifacts found under {args.predictions_dir}."
        )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptp", type=Path, default=Path("checkpoints/ptp_predictions.npz"))
    parser.add_argument("--independent", type=Path, default=Path("checkpoints/independent_predictions.npz"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = discover_artifacts(args)
    full_results = {name: score_file(path) for name, path in artifacts.items()}

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
                "token_accuracy": values["token_accuracy"],
                "tail_token_accuracy": values["tail_token_accuracy"],
                "prefix8_success_rate": values["prefix8_success_rate"],
                "relative_param_multiplier": values["param_count_M"] / baseline_params,
            }
        return public_results

    experiments = {
        "synthetic_auxiliary_ablation": {"results": public(full_results)},
    }

    # N=8 is the scored restricted-compute setting for this gym.
    for proposal_length in (8,):
        truncated = {
            name: score_file(path, proposal_length=proposal_length)
            for name, path in artifacts.items()
        }
        experiments[f"synthetic_restricted_compute_n{proposal_length}"] = {
            "results": public(truncated)
        }

    scores = {"experiments": experiments}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
