# Verification Report

## Initial Assessment
The workspace started at `core_claim_plus`. That milestone is justified for the
scaled synthetic gym: the implemented method is a real parallel token proposal
mechanism with inverse-CDF reference sampling, artifact-based verification, and
a clear same-scale baseline gap. It does not reproduce the full paper-scale
TinyLlama/Vicuna experiments, wall-clock speedups, C-PTP training, or partial
quadratic decoding.

## Issues Found
- Minor: `validate.py --compare` warns that `eval/train/evaluate.py` and
  `eval/test/evaluate.py` regenerate data. Manual inspection shows this is a
  false-positive heuristic: both evaluators read `checkpoints/*_predictions.npz`
  artifacts, verify the fixed auxiliary batch, and only regenerate reference
  targets from `data/`.
- Cosmetic/infrastructure: tracked system files `README.md`, `manifest.json`,
  and `trajectory/turn_*.jsonl` are dirty. Attempting to restore them failed
  because `README.md` and `manifest.json` are exposed as busy character devices
  and `trajectory/` is read-only. These files are outside the reproduction gym
  deliverables and do not affect scripts, scoring, method code, or evaluation.
- Cosmetic: generated Python `__pycache__` directories were present in the
  workspace.

## Fixes Applied
- Removed local generated Python `__pycache__` directories outside the read-only
  shared mount.
- Re-ran `python validate.py --compare`; scoring, import separation,
  train/test method-key consistency, and score discrimination all pass.
- Reviewed the method-agnostic files (`briefing/problem.md`,
  `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`) and found no architecture,
  loss, or implementation-recipe leakage.
- Reviewed scoring coefficients and weights. The primary accepted-prefix metric
  receives the largest coefficient, additional accuracy metrics guard against
  prefix-only gaming, and `relative_param_multiplier` is a negative constraint.
- Reviewed `scripts/download.sh`; no external downloads or `shared/` references
  are required because data is procedural and tracked in `data/`.

## Final Judgment
The workspace is ready to serve as a research gym for the scaled benchmark.
Improve-mode agents can run the packaged method and baseline, while
from-scratch agents can rely on `data/`, `eval/`, and the method-agnostic
briefing/scoring guidance without seeing the paper's implementation. The
evaluation is reusable for any method that writes compatible
`*_predictions.npz` artifacts.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic scaled benchmark rather than the paper's full
  language-model experiments

## Hardening (2026-11-11)
The scoring contract changed after this verification: see `HARDENING.md`.
The scored artifact is now `checkpoints/<method>_proposer.pt`; the hidden test
holds its own control values, teacher and architecture, and the parameter
count is a measured eligibility gate rather than a ranked metric. The
reproduced effect (O-PTP 12.136 vs independent 4.347 accepted tokens on the
hidden test) and the `core_claim_plus` judgment are unchanged.
