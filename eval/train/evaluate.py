from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.ptp_synthetic import fixed_eval_batch


SLICE_NAME = "train"
SPLIT_SEED = 271828


def prefix_lengths(teacher: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matches = teacher == prediction
    lengths = np.zeros(matches.shape[0], dtype=np.int64)
    for i, row in enumerate(matches):
        wrong = np.flatnonzero(~row)
        lengths[i] = int(wrong[0]) if wrong.size else row.size
    return lengths


def split_indices(num_examples: int) -> np.ndarray:
    if num_examples < 2:
        raise ValueError("train/test slicing requires at least two evaluation examples")
    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(num_examples)
    return np.sort(indices[: num_examples // 2])


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
            name = artifact_name(path)
            artifacts.setdefault(name, path)
            seen_paths.add(resolved)

    if not artifacts:
        raise FileNotFoundError(
            f"No prediction artifacts found under {args.predictions_dir}."
        )
    return artifacts


def score_file(path: Path, proposal_length: int | None = None) -> dict[str, float]:
    arr = np.load(path)
    if "auxiliaries" not in arr:
        raise KeyError(
            f"{path} must contain an 'auxiliaries' array; eval regenerates teacher targets"
        )
    auxiliaries = arr["auxiliaries"]
    expected_auxiliaries, teacher = fixed_eval_batch(
        auxiliaries.shape[0], auxiliaries.shape[1]
    )
    if not np.array_equal(auxiliaries, expected_auxiliaries):
        raise ValueError(f"{path} was not produced on the fixed held-out auxiliary batch")

    indices = split_indices(auxiliaries.shape[0])
    teacher = teacher[indices]
    prediction = arr["prediction"][indices]

    if proposal_length is not None:
        teacher = teacher[:, :proposal_length]
        prediction = prediction[:, :proposal_length]

    lengths = prefix_lengths(teacher, prediction)
    meta_path = path.with_suffix(".json")
    param_count = (
        json.loads(meta_path.read_text()).get("param_count", 0)
        if meta_path.exists()
        else 0
    )
    return {
        "avg_correct_tokens": float(lengths.mean()),
        "avg_accepted_tokens": float((lengths + 1).clip(max=teacher.shape[1]).mean()),
        "param_count_M": float(param_count) / 1_000_000.0,
    }


def public(values_by_method: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    if "independent_prediction" in values_by_method:
        baseline_params = max(
            values_by_method["independent_prediction"]["param_count_M"], 1e-12
        )
    else:
        baseline_params = max(next(iter(values_by_method.values()))["param_count_M"], 1e-12)

    public_results = {}
    for name, values in values_by_method.items():
        public_results[name] = {
            "avg_correct_tokens": values["avg_correct_tokens"],
            "avg_accepted_tokens": values["avg_accepted_tokens"],
            "relative_param_multiplier": values["param_count_M"] / baseline_params,
        }
    return public_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptp", type=Path, default=Path("checkpoints/ptp_predictions.npz"))
    parser.add_argument(
        "--independent",
        type=Path,
        default=Path("checkpoints/independent_predictions.npz"),
    )
    parser.add_argument("--predictions-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_train.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = discover_artifacts(args)

    full_results = {name: score_file(path) for name, path in artifacts.items()}
    restricted_results = {
        name: score_file(path, proposal_length=8) for name, path in artifacts.items()
    }

    scores = {
        "experiments": {
            "synthetic_auxiliary_ablation": {"results": public(full_results)},
            "synthetic_restricted_compute_n8": {"results": public(restricted_results)},
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps({"slice": SLICE_NAME, **scores}, indent=2))


if __name__ == "__main__":
    main()
