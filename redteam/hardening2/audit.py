"""Second hardening audit: converged reference, variants and shortcuts.

Same procedure as ``redteam/hardening/audit.py``: every case runs in its own
copy of the gym under tmp/hardening2_ws/<case>; its preparation may edit
anything a method can reach; ``eval/`` and ``scripts/evaluate_test.sh`` are then
restored from the pristine gym and ``bash scripts/evaluate_test.sh`` runs with
no arguments as a fresh process. The shortcut preparations are imported from
the first audit unchanged (they submit against the current reference files).

Commands:
  python -m redteam.hardening2.audit run       # every case whose inputs exist
  python -m redteam.hardening2.audit paired    # paired per-block statistics
  python -m redteam.hardening2.audit rank      # rank table + ladder + seed spread
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HARD = ROOT / "redteam" / "hardening2"
RESULTS = HARD / "results"
ARTIFACTS = HARD / "artifacts"
OLD_ARTIFACTS = ROOT / "redteam" / "hardening" / "artifacts"
WS_ROOT = ROOT / "tmp" / "hardening2_ws"
REF = ROOT / "checkpoints" / "o_ptp_proposer.pt"
BASE = ROOT / "checkpoints" / "independent_prediction_proposer.pt"

sys.path.insert(0, str(ROOT))
from redteam.hardening import audit as first  # noqa: E402  (shortcut preparations)

METRICS = ("avg_accepted_tokens", "avg_correct_tokens", "token_accuracy",
           "tail_token_accuracy", "prefix8_success_rate")
EXPERIMENTS = ("synthetic_auxiliary_ablation", "synthetic_restricted_compute_n8")


def copy_variant(src: Path):
    def prep(ws: Path) -> None:
        shutil.copy2(src, ws / "checkpoints" / "o_ptp_proposer.pt")
    prep.requires = [src, REF, BASE]
    return prep


def new_variant(name: str):
    return copy_variant(ARTIFACTS / name / "o_ptp_proposer.pt")


def prep_reference(ws: Path) -> None:
    return None


prep_reference.requires = [REF, BASE]


# name -> (group, prep, run_train, after_train, description, training steps or None)
CASES: dict[str, tuple] = {
    "reference": ("variant", prep_reference, True, None,
                  "final reference O-PTP: seed 13, 150k steps x 512, cosine", 150_000),
    "seed14": ("variant", new_variant("seed14"), True, None, "reference recipe, seed 14", 150_000),
    "seed15": ("variant", new_variant("seed15"), True, None, "reference recipe, seed 15", 150_000),
    "train_2x": ("variant", new_variant("train_2x"), True, None,
                 "more training alone: 300k steps, seed 13", 300_000),
    "train_2x_seed14": ("variant", new_variant("train_2x_seed14"), True, None,
                        "more training alone: 300k steps, seed 14", 300_000),
    "ladder_75k": ("variant", new_variant("ladder_75k"), True, None, "75k steps", 75_000),
    "ladder_37k": ("variant", new_variant("ladder_37k"), True, None, "37.5k steps", 37_500),
    "ladder_19k": ("variant", new_variant("ladder_19k"), True, None, "18.75k steps", 18_750),
    "ladder_9k": ("variant", new_variant("ladder_9k"), True, None, "9.4k steps", 9_375),
    "arch_4layer_h84": ("variant", new_variant("arch_4layer_h84"), True, None,
                        "better: 4 layers x 84 wide (paper depth), under cap, reference recipe",
                        150_000),
    "ar_nn": ("variant", new_variant("ar_nn"), True, None,
              "[ar+NN] auxiliary embedding, reference recipe", 150_000),
    "cptp": ("variant", new_variant("cptp"), True, None,
             "weaker (paper Tables 6/7): C-PTP, aux offset 1 + inverse-CDF decode", 150_000),
    "th_lin": ("variant", new_variant("th_lin"), True, None,
               "weaker (paper Table 7): [th+lin] auxiliary embedding", 150_000),
    "old_reference_1200": ("variant", copy_variant(ARTIFACTS / "old_reference" / "o_ptp_proposer.pt"),
                           True, None, "weaker: previous reference, 1,200 steps x 192, constant lr",
                           1_200),
    "old_training_3600": ("variant", copy_variant(OLD_ARTIFACTS / "better_training" / "o_ptp_proposer.pt"),
                          True, None, "the review's 3x-training run: 3,600 steps x 192, constant lr",
                          3_600),
    "weaker_no_auxiliary": ("variant", copy_variant(BASE), True, None,
                            "weaker: the trained same-size proposer without auxiliaries", 150_000),
}
for _name in ("visible_answers", "sequential_torchscript", "sequential_pickled_module",
              "sequential_deep_config", "false_params", "inflate_baseline",
              "oversized_proposer", "replace_teacher", "omit_failed_method", "shrink_test",
              "forged_scorecard", "empty_output", "wrong_shapes", "malicious_pickle",
              "import_hijack", "invalid_config"):
    _group, _prep, _train, _after, _desc = first.CASES[f"shortcut_{_name}"]
    CASES[f"shortcut_{_name}"] = ("shortcut", _prep, _train, _after, _desc, None)


# ------------------------------------------------------------------ runner
def make_ws(case: str) -> Path:
    ws = WS_ROOT / case
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    for name in ("eval", "data", "scripts"):
        shutil.copytree(ROOT / name, ws / name, ignore=first.ignore)
    (ws / "scoring").mkdir()
    for path in (ROOT / "scoring").glob("*.json"):
        shutil.copy2(path, ws / "scoring" / path.name)
    (ws / "checkpoints").mkdir()
    shutil.copy2(REF, ws / "checkpoints" / REF.name)
    shutil.copy2(BASE, ws / "checkpoints" / BASE.name)
    return ws


def restore_eval(ws: Path) -> None:
    shutil.rmtree(ws / "eval")
    shutil.copytree(ROOT / "eval", ws / "eval", ignore=first.ignore)
    shutil.copy2(ROOT / "scripts" / "evaluate_test.sh", ws / "scripts" / "evaluate_test.sh")


def command(args: list[str], cwd: Path, log: Path) -> int:
    env = dict(os.environ, HOME=str(cwd))  # a planted ~/.local must also be inert
    with log.open("w") as handle:
        return subprocess.run(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                              env=env).returncode


def run_case(name: str, force: bool = False) -> None:
    group, prep, run_train, after_train, description, _ = CASES[name]
    out = RESULTS / name
    if (out / "status.json").exists() and not force:
        return
    missing = [str(p) for p in getattr(prep, "requires", [REF, BASE]) if not Path(p).exists()]
    if missing:
        print(f"SKIP {name}: missing {missing}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    status: dict = {"group": group, "description": description}
    started = time.monotonic()
    print(f"START {name}", flush=True)
    try:
        ws = make_ws(name)
        prep(ws)
        status["prepared"] = True
        if run_train:
            status["train_rc"] = command(["bash", "scripts/evaluate_train.sh"], ws, out / "train.log")
            if (ws / "scoring" / "scores_train.json").exists():
                shutil.copy2(ws / "scoring" / "scores_train.json", out / "scores_train.json")
        if after_train:
            after_train(ws)
        restore_eval(ws)
        status["test_rc"] = command(["bash", "scripts/evaluate_test.sh"], ws, out / "test.log")
        result = ws / "scoring" / "scores_test.json"
        status["test_output_written"] = result.exists()
        if result.exists():
            shutil.copy2(result, out / "scores_test.json")
        status["code_executed_marker"] = (ws / "PWNED").exists()
        status["import_hijack_marker"] = (ws / "HIJACKED").exists()
    except Exception as exc:  # preparation failures are recorded, not hidden
        status["error"] = repr(exc)
    status["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(f"END {name}: {status}", flush=True)
    shutil.rmtree(WS_ROOT / name, ignore_errors=True)


def cmd_run(force: bool = False) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in CASES:
        run_case(name, force=force)


# ----------------------------------------------------------- paired stats
def per_block(model, aux, teacher, contract, horizon=None):
    tokens = contract.run_proposer(model, aux)
    if horizon:
        tokens, teacher = tokens[:, :horizon], teacher[:, :horizon]
    matches = tokens == teacher
    lengths = np.where(matches.all(1), matches.shape[1], np.argmin(matches, 1))
    accepted = np.minimum(lengths + 1, matches.shape[1]).astype(np.float64)
    return accepted, lengths.astype(np.float64)


def artifact_of(name: str) -> Path | None:
    if name == "reference":
        return REF
    if name == "weaker_no_auxiliary":
        return BASE
    if name == "old_reference_1200":
        return ARTIFACTS / "old_reference" / "o_ptp_proposer.pt"
    if name == "old_training_3600":
        return OLD_ARTIFACTS / "better_training" / "o_ptp_proposer.pt"
    path = ARTIFACTS / name / "o_ptp_proposer.pt"
    return path


def cmd_paired() -> None:
    """Paired per-block differences vs the reference on the hidden inputs."""
    sys.path.insert(0, str(ROOT / "eval" / "test"))
    import evaluate as hidden  # eval/test/evaluate.py (audit-only import)
    import hidden_proposer as contract

    aux = hidden.test_auxiliaries()
    teacher = hidden.sample_teacher_from_u(aux)
    per = {}
    for name, case in CASES.items():
        if case[0] != "variant":
            continue
        path = artifact_of(name)
        if path is None or not path.exists():
            continue
        model, _, _ = contract.load_proposer(path)
        per[name] = {"full": per_block(model, aux, teacher, contract),
                     "n8": per_block(model, aux, teacher, contract, horizon=8)}
    out = {"num_blocks": int(aux.shape[0]), "reference": "reference", "comparisons": {}}
    ref = per["reference"]
    for name, values in per.items():
        row = {}
        for exp in ("full", "n8"):
            for idx, metric in ((0, "accepted"), (1, "correct")):
                diff = values[exp][idx] - ref[exp][idx]
                row[f"{exp}_{metric}_mean"] = float(values[exp][idx].mean())
                row[f"{exp}_{metric}_paired_diff"] = float(diff.mean())
                row[f"{exp}_{metric}_paired_se"] = float(diff.std(ddof=1) / math.sqrt(diff.size))
        out["comparisons"][name] = row
    (HARD / "paired.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# ------------------------------------------------------------------- ranks
def compute_ranks(names, contract, ref, results):
    scales = {}
    for exp, config in contract.items():
        scales[exp] = {}
        for metric in config["metrics"]:
            values = [results[n][exp]["results"]["o_ptp"][metric] for n in names
                      if metric in results[n].get(exp, {}).get("results", {}).get("o_ptp", {})]
            scale = statistics.pstdev(values) if len(values) > 1 else 0.0
            scales[exp][metric] = scale or 1.0
    ranks = {}
    for name in names:
        total = 0.0
        for exp, config in contract.items():
            values = results[name].get(exp, {}).get("results", {}).get("o_ptp", {})
            for metric, meta in config["metrics"].items():
                if metric not in values:
                    continue
                direction = 1.0 if meta["higher_is_better"] else -1.0
                total += (config["weight"] * abs(meta["coefficient"]) * direction
                          * (values[metric] - ref[exp]["results"]["o_ptp"][metric])
                          / scales[exp][metric])
        ranks[name] = total
    return {"scales": scales, "ranks": ranks}


def o_ptp_value(results, name, exp, metric):
    return results[name][exp]["results"]["o_ptp"][metric]


def cmd_rank() -> None:
    reference = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    contract = {exp: {"weight": cfg["weight"], "metrics": cfg["metrics"]}
                for exp, cfg in reference.items()}
    ref = json.loads((ROOT / "scoring" / "scores_test.json").read_text())["experiments"]
    results, train_results, groups, statuses = {}, {}, {}, {}
    for name in CASES:
        path = RESULTS / name / "scores_test.json"
        if path.exists():
            results[name] = json.loads(path.read_text())["experiments"]
            statuses[name] = json.loads((RESULTS / name / "status.json").read_text())
            groups[name] = CASES[name][0]
            train_path = RESULTS / name / "scores_train.json"
            if train_path.exists():
                train_results[name] = json.loads(train_path.read_text())["experiments"]
    honest = [n for n in results if groups[n] == "variant"]
    everyone = [n for n in results if groups[n] in ("variant", "shortcut")]

    seeds = [n for n in ("reference", "seed14", "seed15") if n in results]
    spread = {}
    for exp in EXPERIMENTS:
        spread[exp] = {}
        for metric in METRICS:
            vals = [o_ptp_value(results, n, exp, metric) for n in seeds]
            spread[exp][metric] = {"min": min(vals), "max": max(vals),
                                   "range": max(vals) - min(vals), "n": len(vals)}
    honest_ranks = compute_ranks(honest, contract, ref, results)
    seed_ranks = [honest_ranks["ranks"][n] for n in seeds]
    ladder = []
    for name in ("ladder_9k", "ladder_19k", "ladder_37k", "ladder_75k", "reference", "train_2x"):
        if name in results:
            ladder.append({
                "case": name, "steps": CASES[name][5],
                "full_accepted": o_ptp_value(results, name, EXPERIMENTS[0], "avg_accepted_tokens"),
                "full_token_accuracy": o_ptp_value(results, name, EXPERIMENTS[0], "token_accuracy"),
                "n8_correct": o_ptp_value(results, name, EXPERIMENTS[1], "avg_correct_tokens"),
                "rank_honest": honest_ranks["ranks"][name],
            })
    agreement = {}
    for name in honest:
        if name in train_results:
            agreement[name] = {
                "train_full_accepted": train_results[name][EXPERIMENTS[0]]["results"]["o_ptp"]["avg_accepted_tokens"],
                "test_full_accepted": o_ptp_value(results, name, EXPERIMENTS[0], "avg_accepted_tokens"),
                "train_n8_correct": train_results[name][EXPERIMENTS[1]]["results"]["o_ptp"]["avg_correct_tokens"],
                "test_n8_correct": o_ptp_value(results, name, EXPERIMENTS[1], "avg_correct_tokens"),
            }
    table = {
        "formula": "sum over experiments and metrics of weight*|coefficient|*direction*"
                   "(value - reference)/pstdev(population), o_ptp entries, reference = "
                   "scoring/scores_test.json",
        "population_honest": honest_ranks,
        "population_all": compute_ranks(everyone, contract, ref, results),
        "seed_spread": spread,
        "seed_rank_range": (max(seed_ranks) - min(seed_ranks)) if seed_ranks else None,
        "ladder": ladder,
        "train_test_agreement": agreement,
        "raw": {n: {exp: results[n][exp]["results"].get("o_ptp") for exp in contract}
                for n in results},
        "status": statuses,
    }
    (HARD / "ranks.json").write_text(json.dumps(table, indent=2))
    lines = [f"{'case':36s} {'full acc':>9s} {'n8 corr':>8s} {'rank(honest)':>13s} {'rank(all)':>10s}"]
    for name in results:
        full = results[name][EXPERIMENTS[0]]["results"].get("o_ptp", {})
        n8 = results[name][EXPERIMENTS[1]]["results"].get("o_ptp", {})
        rh = table["population_honest"]["ranks"].get(name)
        ra = table["population_all"]["ranks"].get(name)
        lines.append(f"{name:36s} {full.get('avg_accepted_tokens', float('nan')):9.4f} "
                     f"{n8.get('avg_correct_tokens', float('nan')):8.4f} "
                     f"{'' if rh is None else f'{rh:+.4f}':>13s} {'' if ra is None else f'{ra:+.4f}':>10s}")
    lines.append("")
    lines.append(f"seed rank range (reference/seed14/seed15): {table['seed_rank_range']}")
    for exp in EXPERIMENTS:
        for metric in ("avg_accepted_tokens", "avg_correct_tokens"):
            s = spread[exp][metric]
            lines.append(f"seed spread {exp} {metric}: {s['min']:.4f}..{s['max']:.4f} (range {s['range']:.4f})")
    lines.append("")
    lines.append("ladder (seed 13, same recipe, cosine over the stated steps):")
    for row in ladder:
        lines.append(f"  {row['steps']:>7d} steps: full accepted {row['full_accepted']:.4f}  "
                     f"token acc {row['full_token_accuracy']:.4f}  n8 correct {row['n8_correct']:.4f}  "
                     f"rank {row['rank_honest']:+.4f}")
    (HARD / "ranks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    command_name = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command_name == "run":
        cmd_run(force="--force" in sys.argv)
    elif command_name == "paired":
        cmd_paired()
    elif command_name == "rank":
        cmd_rank()
    else:
        raise SystemExit(f"unknown command {command_name}")


if __name__ == "__main__":
    main()
