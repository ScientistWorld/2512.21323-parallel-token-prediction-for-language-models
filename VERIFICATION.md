# Verification Report

## Initial Assessment
The workspace already contained a working scaled reproduction at
`core_claim_plus`: a proposed block predictor beats an independent same-scale
baseline on accepted-prefix metrics, with train/test split score files present.
The claimed milestone is justified for the scaled synthetic gym, not for the
paper's full TinyLlama/Vicuna wall-clock experiments.

## Issues Found
- Minor: method-agnostic guidance used paper/method-specific wording and the
  top-level evaluation briefing misstated the artifact contract.
- Minor: `scripts/evaluate.sh` only recognized the two original artifact names,
  which made it less reusable for future methods.
- Minor: `reference.json` included derived accepted/correct values without
  explicitly documenting how they were derived from paper-reported metrics.
- Cosmetic: an ignored root crash dump was present and was removed.

## Fixes Applied
- Rewrote `briefing/problem.md`, `briefing/evaluation.md`, and the scoring
  guidance files to describe the problem, constraints, and direction without
  exposing architecture, losses, or implementation details.
- Updated `eval/evaluate_ptp.py` to discover any `*_predictions.npz` artifact in
  `checkpoints/`, while preserving aliases for the reproduced method and
  baseline.
- Added experiment-level notes in `scoring/reference.json` explaining derived
  metrics.
- Ran `python validate.py --compare` and `py_compile` checks successfully.

## Final Judgment
The workspace is ready to serve as a research gym for the scaled benchmark. It
has clean problem/solution separation, reusable artifact-based evaluation, a
nontrivial baseline gap, constraint metrics for parameter count, and matching
train/test score artifacts. The main limitation is that the gym uses a compact
synthetic reference generator rather than the paper's full language-model
benchmarks.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic data only
