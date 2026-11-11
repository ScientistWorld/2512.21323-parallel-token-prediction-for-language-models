"""Train-slice (development) evaluator for the parallel block-proposal benchmark.

Same protocol and code path as the hidden test, on a public control batch
(seed 1013, 4096 blocks; the same batch ``data.ptp_synthetic.fixed_eval_batch``
returns). ``bash scripts/evaluate_train.sh`` scores every
``checkpoints/<name>_proposer.pt`` artifact on disk:

* control values: the public development batch;
* targets: this directory's frozen copy of the reference teacher;
* proposer: rebuilt from this directory's frozen contract, tensors loaded with
  ``weights_only=True``, parameters counted here, one forward call per block;
* nothing the method wrote (scores, sidecars, predictions, flags) is read.

A missing, unloadable, out-of-contract, over-cap or non-finite proposer is a
failed run and receives the worst value of every scored metric in every
experiment. The canonical slots ``o_ptp`` and ``independent_prediction`` are
always written, failed if absent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import dev_proposer as contract  # noqa: E402
from dev_teacher import SEQ_LEN, sample_teacher_from_u  # noqa: E402


SLICE_NAME = "train"
DEV_SEED = 1013
NUM_EXAMPLES = 4096
RESTRICTED_HORIZON = 8
REQUIRED_SLOTS = ("independent_prediction", "o_ptp")
ARTIFACT_SUFFIX = "_proposer.pt"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
FULL_EXPERIMENT = "synthetic_auxiliary_ablation"
RESTRICTED_EXPERIMENT = "synthetic_restricted_compute_n8"
METRICS = (
    "avg_accepted_tokens",
    "avg_correct_tokens",
    "token_accuracy",
    "tail_token_accuracy",
    "prefix8_success_rate",
)


def worst_values() -> dict[str, float]:
    # #accepted >= 1 always (the verifier's free token); every other metric >= 0.
    return {
        "avg_accepted_tokens": 1.0,
        "avg_correct_tokens": 0.0,
        "token_accuracy": 0.0,
        "tail_token_accuracy": 0.0,
        "prefix8_success_rate": 0.0,
    }


def dev_auxiliaries() -> np.ndarray:
    rng = np.random.default_rng(DEV_SEED)
    return rng.random((NUM_EXAMPLES, SEQ_LEN), dtype=np.float32)


def prefix_lengths(teacher: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matches = teacher == prediction
    first_wrong = np.argmin(matches, axis=1)
    return np.where(matches.all(axis=1), matches.shape[1], first_wrong)


def block_metrics(teacher: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    horizon = teacher.shape[1]
    lengths = prefix_lengths(teacher, prediction)
    midpoint = max(horizon // 2, 1)
    first8 = min(8, horizon)
    values = {
        "avg_accepted_tokens": float(np.minimum(lengths + 1, horizon).mean()),
        "avg_correct_tokens": float(lengths.mean()),
        "token_accuracy": float((teacher == prediction).mean()),
        "tail_token_accuracy": float((teacher[:, midpoint:] == prediction[:, midpoint:]).mean()),
        "prefix8_success_rate": float(
            np.all(teacher[:, :first8] == prediction[:, :first8], axis=1).mean()
        ),
    }
    bounds = {
        "avg_accepted_tokens": (1.0, float(horizon)),
        "avg_correct_tokens": (0.0, float(horizon)),
    }
    for key, value in values.items():
        low, high = bounds.get(key, (0.0, 1.0))
        if not (np.isfinite(value) and low <= value <= high):
            raise contract.ProposerContractError(f"impossible {key}={value}")
    return values


def discover(checkpoints: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if checkpoints.is_dir():
        for path in sorted(checkpoints.iterdir()):
            name = path.name[: -len(ARTIFACT_SUFFIX)]
            if path.name.endswith(ARTIFACT_SUFFIX) and name:
                found[name] = path
    return found


def score_artifact(path: Path | None, auxiliaries: np.ndarray, teacher: np.ndarray):
    if path is None:
        raise contract.ProposerContractError("no proposer artifact on disk")
    model, config, param_count = contract.load_proposer(path)
    tokens = contract.run_proposer(model, auxiliaries)
    if tokens.shape != teacher.shape:
        raise contract.ProposerContractError("prediction shape mismatch")
    if tokens.min() < 0 or tokens.max() >= contract.VOCAB_SIZE:
        raise contract.ProposerContractError("token outside the vocabulary")
    full = block_metrics(teacher, tokens)
    restricted = block_metrics(
        teacher[:, :RESTRICTED_HORIZON], tokens[:, :RESTRICTED_HORIZON]
    )
    return full, restricted, config, param_count


def parse_args() -> argparse.Namespace:
    root = HERE.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--output", type=Path, default=root / "scoring" / "scores_train.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = args.checkpoints_dir
    output = args.output

    auxiliaries = dev_auxiliaries()
    teacher = sample_teacher_from_u(auxiliaries)
    artifacts = discover(checkpoints)
    names = sorted(set(artifacts) | set(REQUIRED_SLOTS))

    full_results: dict[str, dict[str, float]] = {}
    restricted_results: dict[str, dict[str, float]] = {}
    constraints: dict[str, dict] = {}
    for name in names:
        record: dict = {
            "param_cap": contract.MAX_PARAM_COUNT,
            "baseline_param_count": contract.BASELINE_PARAM_COUNT,
        }
        try:
            if not NAME_RE.match(name):
                raise contract.ProposerContractError("invalid method name")
            full, restricted, config, param_count = score_artifact(
                artifacts.get(name), auxiliaries, teacher
            )
            record.update(
                status="ok",
                param_count=param_count,
                relative_param_multiplier=param_count / contract.BASELINE_PARAM_COUNT,
                config=config,
            )
        except Exception as exc:  # any failure is a failed run, never a crash
            full, restricted = worst_values(), worst_values()
            kind = "constraint" if isinstance(exc, contract.ProposerConstraintError) else "contract"
            record.update(status="failed", failure=kind, reason=str(exc)[:500])
        full_results[name] = full
        restricted_results[name] = restricted
        constraints[name] = record
        print(f"[train] {name}: {record['status']} {full['avg_accepted_tokens']:.4f}", flush=True)

    scores = {
        "slice": SLICE_NAME,
        "experiments": {
            FULL_EXPERIMENT: {"results": full_results},
            RESTRICTED_EXPERIMENT: {"results": restricted_results},
        },
        "_constraints": constraints,
        "_protocol": {
            "num_examples": NUM_EXAMPLES,
            "seq_len": SEQ_LEN,
            "restricted_horizon": RESTRICTED_HORIZON,
            "artifact_pattern": f"checkpoints/*{ARTIFACT_SUFFIX}",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(scores, indent=2))
    os.replace(tmp, output)
    print(json.dumps(scores["experiments"], indent=2))


if __name__ == "__main__":
    main()
