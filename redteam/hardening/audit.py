"""Hardening audit: honest variants and shortcuts through the fixed hidden test.

Every case runs in its own copy of the gym under tmp/hardening_ws/<case>:
the case's preparation may edit anything a method can reach (checkpoints/,
data/, scoring/, the workspace root); eval/ and scripts/evaluate_test.sh are
then restored from the pristine gym, and ``bash scripts/evaluate_test.sh`` runs
with no arguments as a fresh process. The real workspace is never touched.

Commands:
  python -m redteam.hardening.audit run      # every case whose inputs exist
  python -m redteam.hardening.audit paired   # paired per-block comparisons
  python -m redteam.hardening.audit rank     # rank table (audit formula)
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
import torch

ROOT = Path(__file__).resolve().parents[2]
HARD = ROOT / "redteam" / "hardening"
RESULTS = HARD / "results"
ARTIFACTS = HARD / "artifacts"
WS_ROOT = ROOT / "tmp" / "hardening_ws"
REF = ROOT / "checkpoints" / "o_ptp_proposer.pt"
BASE = ROOT / "checkpoints" / "independent_prediction_proposer.pt"
LEGACY = ROOT / "checkpoints" / "legacy"

sys.path.insert(0, str(ROOT))
from data.proposer import FORMAT, ProposerNet, save_proposer  # noqa: E402
from data.ptp_synthetic import cdf_table, fixed_eval_batch  # noqa: E402

O_PTP_CONFIG = {
    "hidden_size": 96, "layers": 3, "heads": 4, "ff_mult": 4,
    "aux_embedding": "ar", "aux_head": "lin", "aux_offset": 0, "decode": "argmax",
}
INDEPENDENT_CONFIG = dict(O_PTP_CONFIG, aux_embedding="none")
PERFECT = {
    "avg_accepted_tokens": 32.0, "avg_correct_tokens": 32.0, "token_accuracy": 1.0,
    "tail_token_accuracy": 1.0, "prefix8_success_rate": 1.0,
}


# ----------------------------------------------------------------- helpers
def o_ptp(ws: Path) -> Path:
    return ws / "checkpoints" / "o_ptp_proposer.pt"


def convert_legacy(src: Path, dst: Path) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=True)
    use_aux = bool(payload["config"]["use_auxiliary"])
    config = dict(O_PTP_CONFIG if use_aux else INDEPENDENT_CONFIG)
    torch.save({"format": FORMAT, "config": config, "state_dict": payload["state_dict"]}, dst)


def legacy_variant(src: Path):
    def prep(ws: Path) -> None:
        convert_legacy(src, o_ptp(ws))
    prep.requires = [src]
    return prep


def final_variant(name: str):
    src = ARTIFACTS / name / "o_ptp_proposer.pt"

    def prep(ws: Path) -> None:
        shutil.copy2(src, o_ptp(ws))
    prep.requires = [src]
    return prep


def constant_proposer(token: int) -> ProposerNet:
    model = ProposerNet(O_PTP_CONFIG)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[token] = 10.0
    return model


# ------------------------------------------------------------- preparations
def prep_reference(ws: Path) -> None:
    return None


def prep_no_auxiliary(ws: Path) -> None:
    # Trained, same-size proposer with the key idea removed (the baseline).
    shutil.copy2(BASE, o_ptp(ws))


def prep_visible_answers(ws: Path) -> None:
    # Reconstruct every public target with the visible generator and submit
    # them as predictions; no proposer is submitted.
    o_ptp(ws).unlink()
    for n in (4096, 16384):
        us, teacher = fixed_eval_batch(n)
        np.savez_compressed(ws / "checkpoints" / f"o_ptp_predictions_{n}.npz",
                            auxiliaries=us, prediction=teacher, teacher=teacher)
    us, teacher = fixed_eval_batch(4096)
    np.savez_compressed(ws / "checkpoints" / "ptp_predictions.npz",
                        auxiliaries=us, prediction=teacher, teacher=teacher)


class SequentialTeacher(torch.nn.Module):
    """Ordinary autoregressive decoding with the public teacher (T calls)."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch = u.shape[0]
        steps = u.shape[1]
        tokens = torch.zeros(batch, steps, dtype=torch.long)
        zeros = torch.zeros(batch, dtype=torch.long)
        for k in range(steps):
            prev = zeros
            prev2 = zeros
            if k >= 1:
                prev = tokens[:, k - 1]
            if k >= 2:
                prev2 = tokens[:, k - 2]
            cdf = self.table[k][prev, prev2]
            picked = (cdf <= u[:, k].double().unsqueeze(1)).sum(1).clamp(max=15)
            tokens[:, k] = picked
        return torch.nn.functional.one_hot(tokens, 16).float() * 10.0


def prep_sequential_torchscript(ws: Path) -> None:
    module = SequentialTeacher(torch.from_numpy(np.array(cdf_table())))
    torch.jit.save(torch.jit.script(module), o_ptp(ws))


def prep_sequential_pickled_module(ws: Path) -> None:
    module = SequentialTeacher(torch.from_numpy(np.array(cdf_table())))
    torch.save({"format": FORMAT, "config": O_PTP_CONFIG, "module": module}, o_ptp(ws))


def prep_sequential_deep_config(ws: Path) -> None:
    # Ask the evaluator for one layer per position (sequential depth).
    config = dict(O_PTP_CONFIG, hidden_size=32, layers=32, heads=4, ff_mult=1)
    torch.save({"format": FORMAT, "config": config, "state_dict": {}}, o_ptp(ws))


def prep_false_params(ws: Path) -> None:
    sidecar = {"param_count": 0, "relative_param_multiplier": 0.0}
    (ws / "checkpoints" / "o_ptp_proposer.json").write_text(json.dumps(sidecar))
    (ws / "checkpoints" / "ptp_predictions.json").write_text(json.dumps(sidecar))


def prep_inflate_baseline(ws: Path) -> None:
    big = ProposerNet(dict(INDEPENDENT_CONFIG, hidden_size=512, layers=4, heads=8))
    save_proposer(ws / "checkpoints" / "independent_prediction_proposer.pt", big)
    (ws / "checkpoints" / "independent_prediction_proposer.json").write_text(
        json.dumps({"param_count": 10**12}))


def prep_oversized_proposer(ws: Path) -> None:
    torch.manual_seed(0)
    save_proposer(o_ptp(ws), ProposerNet(dict(O_PTP_CONFIG, hidden_size=128, layers=4)))


def prep_replace_teacher(ws: Path) -> None:
    data = ws / "data" / "ptp_synthetic.py"
    text = data.read_text()
    text = text.replace(
        'def teacher_probs(prefix: np.ndarray, position: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:\n',
        'def teacher_probs(prefix: np.ndarray, position: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:\n'
        '    probs = np.zeros(vocab_size); probs[3] = 1.0; return probs\n',
        1,
    )
    data.write_text(text)
    save_proposer(o_ptp(ws), constant_proposer(3))


def prep_omit_failed_method(ws: Path) -> None:
    model = ProposerNet(O_PTP_CONFIG)
    state = {k: v.clone() for k, v in torch.load(REF, weights_only=True)["state_dict"].items()}
    state["head.weight"][0, 0] = float("nan")
    torch.save({"format": FORMAT, "config": O_PTP_CONFIG, "state_dict": state}, o_ptp(ws))
    del model


def after_omit_failed_method(ws: Path) -> None:
    path = ws / "scoring" / "scores_train.json"
    data = json.loads(path.read_text())
    for exp in data.get("experiments", {}).values():
        exp.get("results", {}).pop("o_ptp", None)
    data.pop("_constraints", None)
    path.write_text(json.dumps(data, indent=2))


def prep_shrink_test(ws: Path) -> None:
    us, teacher = fixed_eval_batch(8)
    np.savez_compressed(ws / "checkpoints" / "o_ptp_predictions.npz",
                        auxiliaries=us, prediction=teacher)
    (ws / "checkpoints" / "o_ptp_proposer.json").write_text(
        json.dumps({"eval_samples": 8, "num_examples": 8}))


def prep_forged_scorecard(ws: Path) -> None:
    for path in (ws / "checkpoints").glob("*_proposer.pt"):
        path.unlink()
    forged = {"slice": "test", "experiments": {
        "synthetic_auxiliary_ablation": {"results": {"o_ptp": PERFECT}},
        "synthetic_restricted_compute_n8": {"results": {"o_ptp": dict(
            PERFECT, avg_accepted_tokens=8.0, avg_correct_tokens=8.0)}}}}
    for name in ("scores_test.json", "scores_train.json", "scores.json"):
        (ws / "scoring" / name).write_text(json.dumps(forged))


def prep_empty_output(ws: Path) -> None:
    for path in (ws / "checkpoints").glob("*_proposer.pt"):
        path.unlink()


def prep_wrong_shapes(ws: Path) -> None:
    state = {k: v.clone() for k, v in torch.load(REF, weights_only=True)["state_dict"].items()}
    state["head.weight"] = torch.zeros(32, 96)
    state["head.bias"] = torch.zeros(32)
    torch.save({"format": FORMAT, "config": O_PTP_CONFIG, "state_dict": state}, o_ptp(ws))


class _Payload:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def prep_malicious_pickle(ws: Path) -> None:
    torch.save({"format": FORMAT, "config": O_PTP_CONFIG,
                "state_dict": _Payload(str(ws / "PWNED"))}, o_ptp(ws))


def prep_import_hijack(ws: Path) -> None:
    forged = {"slice": "test", "experiments": {
        "synthetic_auxiliary_ablation": {"results": {"o_ptp": PERFECT}},
        "synthetic_restricted_compute_n8": {"results": {"o_ptp": PERFECT}}}}
    payload = (
        "import json, os, pathlib\n"
        f"pathlib.Path({str(ws / 'scoring' / 'scores_test.json')!r}).write_text({json.dumps(json.dumps(forged))})\n"
        f"pathlib.Path({str(ws / 'HIJACKED')!r}).touch()\n"
        "os._exit(0)\n"
    )
    for name in ("numpy.py", "torch.py", "sitecustomize.py", "hidden_proposer.py",
                 "hidden_teacher.py", "usercustomize.py"):
        (ws / name).write_text(payload)
    site = ws / ".local" / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (site / "numpy.py").write_text(payload)


def prep_invalid_config(ws: Path) -> None:
    config = dict(O_PTP_CONFIG, hidden_size=-96)
    torch.save({"format": FORMAT, "config": config, "state_dict": {}}, o_ptp(ws))


def prep_smoke_cptp_th_nn(ws: Path) -> None:
    torch.manual_seed(1)
    save_proposer(o_ptp(ws), ProposerNet(dict(
        O_PTP_CONFIG, aux_embedding="th", aux_head="nn", aux_offset=1, decode="pick")))


def prep_smoke_fl(ws: Path) -> None:
    torch.manual_seed(2)
    save_proposer(o_ptp(ws), ProposerNet(dict(O_PTP_CONFIG, aux_embedding="fl")))


for _prep in (prep_reference, prep_visible_answers, prep_sequential_torchscript,
              prep_sequential_pickled_module, prep_sequential_deep_config,
              prep_false_params, prep_inflate_baseline, prep_oversized_proposer,
              prep_replace_teacher, prep_shrink_test, prep_forged_scorecard,
              prep_empty_output, prep_import_hijack, prep_invalid_config,
              prep_smoke_cptp_th_nn, prep_smoke_fl):
    _prep.requires = [REF, BASE]
prep_no_auxiliary.requires = [REF, BASE]
prep_omit_failed_method.requires = [REF, BASE]
prep_wrong_shapes.requires = [REF, BASE]
prep_malicious_pickle.requires = [REF, BASE]

# name -> (group, prep, run_train, after_train, description)
CASES = {
    "reference": ("variant", prep_reference, True, None,
                  "final reference O-PTP (seed 13, 1200 steps, batch 192) from final code"),
    "shipped_original": ("variant", legacy_variant(LEGACY / "ptp_model.pt"), True, None,
                         "the reference shipped before hardening (unknown seed)"),
    "seed13_old": ("variant", legacy_variant(ROOT / "redteam/seed13/ptp_model.pt"), True, None,
                   "redteam rerun, seed 13, same recipe"),
    "seed14_old": ("variant", legacy_variant(ROOT / "redteam/seed14/ptp_model.pt"), True, None,
                   "redteam rerun, seed 14, same recipe"),
    "seed15_old": ("variant", legacy_variant(ROOT / "redteam/seed15/ptp_model.pt"), True, None,
                   "redteam rerun, seed 15, same recipe"),
    "better_training_old": ("variant", legacy_variant(ROOT / "redteam/better_training/ptp_model.pt"),
                            True, None, "redteam: same model, 3600 steps"),
    "weaker_training_old": ("variant", legacy_variant(ROOT / "redteam/weaker_training/ptp_model.pt"),
                            True, None, "redteam: same model, 120 steps"),
    "seed14": ("variant", final_variant("seed14"), True, None, "final code, seed 14"),
    "seed15": ("variant", final_variant("seed15"), True, None, "final code, seed 15"),
    "better_training": ("variant", final_variant("better_training"), True, None,
                        "final code, same model, 3600 steps"),
    "weaker_training": ("variant", final_variant("weaker_training"), True, None,
                        "final code, same model, 120 steps"),
    "arch_4layer_h84": ("variant", final_variant("arch_4layer_h84"), True, None,
                        "final code, 4 layers x 84 wide (paper's depth), 1200 steps, under cap"),
    "weaker_no_auxiliary": ("variant", prep_no_auxiliary, True, None,
                            "trained same-size proposer without auxiliaries"),
    "shortcut_visible_answers": ("shortcut", prep_visible_answers, False, None,
                                 "public generator reconstructs targets; predictions submitted"),
    "shortcut_sequential_torchscript": ("shortcut", prep_sequential_torchscript, False, None,
                                        "TorchScript module running the teacher autoregressively"),
    "shortcut_sequential_pickled_module": ("shortcut", prep_sequential_pickled_module, False, None,
                                           "pickled nn.Module running the teacher autoregressively"),
    "shortcut_sequential_deep_config": ("shortcut", prep_sequential_deep_config, False, None,
                                        "config asking for 32 layers (one per position)"),
    "shortcut_false_params": ("shortcut", prep_false_params, False, None,
                              "reference weights, sidecars claim zero parameters"),
    "shortcut_inflate_baseline": ("shortcut", prep_inflate_baseline, False, None,
                                  "baseline replaced by a huge model claiming 1e12 parameters"),
    "shortcut_oversized_proposer": ("shortcut", prep_oversized_proposer, False, None,
                                    "proposer above the 1.10x parameter cap"),
    "shortcut_replace_teacher": ("shortcut", prep_replace_teacher, False, None,
                                 "public teacher edited to a point mass; constant proposer"),
    "shortcut_omit_failed_method": ("shortcut", prep_omit_failed_method, True, after_omit_failed_method,
                                    "NaN proposer, then its name removed from scores_train.json"),
    "shortcut_shrink_test": ("shortcut", prep_shrink_test, False, None,
                             "reference weights plus 8-example predictions and size sidecar"),
    "shortcut_forged_scorecard": ("shortcut", prep_forged_scorecard, False, None,
                                  "no artifacts; perfect score files pre-written"),
    "shortcut_empty_output": ("shortcut", prep_empty_output, False, None, "no artifacts at all"),
    "shortcut_wrong_shapes": ("shortcut", prep_wrong_shapes, False, None,
                              "state dict whose head does not match the config"),
    "shortcut_malicious_pickle": ("shortcut", prep_malicious_pickle, False, None,
                                  "artifact whose unpickling would run a shell command"),
    "shortcut_import_hijack": ("shortcut", prep_import_hijack, False, None,
                               "planted numpy/torch/sitecustomize modules that forge scores"),
    "shortcut_invalid_config": ("shortcut", prep_invalid_config, False, None,
                                "negative hidden size"),
    "smoke_cptp_th_nn": ("smoke", prep_smoke_cptp_th_nn, False, None,
                         "untrained th+nn embedding, offset 1, inverse-CDF decode"),
    "smoke_fl": ("smoke", prep_smoke_fl, False, None, "untrained scalar embedding"),
}


# ------------------------------------------------------------------ runner
def ignore(_dir, names):
    return [n for n in names if n in ("__pycache__", "downloads")]


def make_ws(case: str) -> Path:
    ws = WS_ROOT / case
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    for name in ("eval", "data", "scripts"):
        shutil.copytree(ROOT / name, ws / name, ignore=ignore)
    (ws / "scoring").mkdir()
    for path in (ROOT / "scoring").glob("*.json"):
        shutil.copy2(path, ws / "scoring" / path.name)
    (ws / "checkpoints").mkdir()
    shutil.copy2(REF, ws / "checkpoints" / REF.name)
    shutil.copy2(BASE, ws / "checkpoints" / BASE.name)
    return ws


def restore_eval(ws: Path) -> None:
    shutil.rmtree(ws / "eval")
    shutil.copytree(ROOT / "eval", ws / "eval", ignore=ignore)
    shutil.copy2(ROOT / "scripts" / "evaluate_test.sh", ws / "scripts" / "evaluate_test.sh")


def command(args: list[str], cwd: Path, log: Path) -> int:
    env = dict(os.environ, HOME=str(cwd))  # a planted ~/.local must also be inert
    with log.open("w") as handle:
        return subprocess.run(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                              env=env).returncode


def run_case(name: str, force: bool = False) -> None:
    group, prep, run_train, after_train, description = CASES[name]
    out = RESULTS / name
    if (out / "status.json").exists() and not force:
        return
    missing = [str(p) for p in getattr(prep, "requires", []) if not Path(p).exists()]
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
def accepted_per_block(model, aux, teacher, contract, horizon=None):
    tokens = contract.run_proposer(model, aux)
    if horizon:
        tokens, teacher = tokens[:, :horizon], teacher[:, :horizon]
    matches = tokens == teacher
    lengths = np.where(matches.all(1), matches.shape[1], np.argmin(matches, 1))
    return np.minimum(lengths + 1, matches.shape[1]).astype(np.float64), lengths.astype(np.float64)


def cmd_paired() -> None:
    """Paired per-block differences vs the reference on the hidden inputs."""
    sys.path.insert(0, str(ROOT / "eval" / "test"))
    import evaluate as hidden  # eval/test/evaluate.py (audit-only import)
    import hidden_proposer as contract

    aux = hidden.test_auxiliaries()
    teacher = hidden.sample_teacher_from_u(aux)
    sources = {"reference": REF, "weaker_no_auxiliary": BASE}
    for name in ("seed14", "seed15", "better_training", "weaker_training", "arch_4layer_h84"):
        sources[name] = ARTIFACTS / name / "o_ptp_proposer.pt"
    tmp = ROOT / "tmp" / "hardening_paired"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, src in (("seed13_old", "redteam/seed13/ptp_model.pt"),
                      ("seed14_old", "redteam/seed14/ptp_model.pt"),
                      ("seed15_old", "redteam/seed15/ptp_model.pt"),
                      ("better_training_old", "redteam/better_training/ptp_model.pt"),
                      ("weaker_training_old", "redteam/weaker_training/ptp_model.pt"),
                      ("shipped_original", "checkpoints/legacy/ptp_model.pt")):
        if (ROOT / src).exists():
            convert_legacy(ROOT / src, tmp / f"{name}.pt")
            sources[name] = tmp / f"{name}.pt"
    per = {}
    for name, path in sources.items():
        if not Path(path).exists():
            continue
        model, _, _ = contract.load_proposer(path)
        per[name] = {
            "full": accepted_per_block(model, aux, teacher, contract),
            "n8": accepted_per_block(model, aux, teacher, contract, horizon=8),
        }
    out = {"num_blocks": int(aux.shape[0]), "comparisons": {}}
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
def load_reference_scores() -> dict:
    return json.loads((ROOT / "scoring" / "scores_test.json").read_text())["experiments"]


def compute_ranks(names: list[str], contract: dict, ref: dict, results: dict) -> dict:
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


def cmd_rank() -> None:
    reference = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    contract = {exp: {"weight": cfg["weight"], "metrics": cfg["metrics"]}
                for exp, cfg in reference.items()}
    ref = load_reference_scores()
    results, groups, statuses = {}, {}, {}
    for name in CASES:
        path = RESULTS / name / "scores_test.json"
        if path.exists():
            results[name] = json.loads(path.read_text())["experiments"]
            statuses[name] = json.loads((RESULTS / name / "status.json").read_text())
            groups[name] = CASES[name][0]
    honest = [n for n in results if groups[n] == "variant"]
    everyone = [n for n in results if groups[n] in ("variant", "shortcut")]
    table = {
        "formula": "sum over experiments and metrics of weight*|coefficient|*direction*"
                   "(value - reference)/pstdev(population), o_ptp entries, reference = "
                   "scoring/scores_test.json",
        "population_honest": compute_ranks(honest, contract, ref, results),
        "population_all": compute_ranks(everyone, contract, ref, results),
        "raw": {n: {exp: results[n][exp]["results"].get("o_ptp") for exp in contract}
                for n in results},
        "status": statuses,
    }
    (HARD / "ranks.json").write_text(json.dumps(table, indent=2))
    lines = [f"{'case':38s} {'full acc':>9s} {'n8 corr':>8s} {'rank(honest)':>13s} {'rank(all)':>10s}"]
    for name in results:
        full = results[name]["synthetic_auxiliary_ablation"]["results"].get("o_ptp", {})
        n8 = results[name]["synthetic_restricted_compute_n8"]["results"].get("o_ptp", {})
        rh = table["population_honest"]["ranks"].get(name)
        ra = table["population_all"]["ranks"].get(name)
        lines.append(f"{name:38s} {full.get('avg_accepted_tokens', float('nan')):9.4f} "
                     f"{n8.get('avg_correct_tokens', float('nan')):8.4f} "
                     f"{'' if rh is None else f'{rh:+.4f}':>13s} {'' if ra is None else f'{ra:+.4f}':>10s}")
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
