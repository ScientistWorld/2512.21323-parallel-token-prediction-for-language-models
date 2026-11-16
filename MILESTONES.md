# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

### [2026-06-01 07:03] - core_claim_plus
- Completed a packaging audit of the scorecard and found the main
  `scoring/scores.json` omitted secondary metrics declared in
  `scoring/reference.json`.
- Updated the evaluator and score artifact so the official scorecard includes
  token accuracy, tail-token accuracy, prefix-8 success rate, and the
  parameter-count constraint.
- Submitting a CPU test job to rerun `scripts/evaluate.sh` inside the managed
  container because the login-node Python environment lacks `numpy`.

### [2026-06-01 07:08] - core_claim_plus
- Final verification confirmed the scaled O-PTP reproduction supports the core
  accepted-prefix claim and the restricted N=8 setting.
- Tightened method-agnostic documentation and made the main evaluator discover
  arbitrary compatible prediction artifacts under `checkpoints/`.
- Documented derived reference metrics and completed `VERIFICATION.md`.

### [2026-06-01 07:14] - core_claim_plus
- Sub-agent audit confirmed the scored path is a real scaled O-PTP
  implementation: inverse-CDF teacher sampling, auxiliary-conditioned causal
  one-call prediction, and teacher-prefix verification.
- Kept the arithmetic-coding `[ar+lin]` auxiliary embedding because Appendix E
  identifies it as the paper's selected structured embedding for experiments.
- Updated `scripts/run.sh` to regenerate baseline and method artifacts before
  scoring, then submitting a CPU test job to validate the fresh-run path.

### [2026-06-01 07:16] - core_claim_plus
- Completed final verification of problem/solution separation, scoring
  coefficients, reusable artifact-based evaluation, train/test split wiring,
  download portability, and milestone consistency.
- Confirmed `python validate.py --compare` passes; the remaining validate
  warnings are false positives because train/test evaluators read prediction
  artifacts and only regenerate reference targets from `data/`.
- Recorded a final `VERIFICATION.md` judgment: ready for gym use at
  `core_claim_plus`, limited to the scaled synthetic benchmark.

### [2026-11-11 08:00] - core_claim_plus (hardening)
- Hidden test now runs each submitted proposer itself (evaluator-owned one-call
  causal-transformer contract, weights-only loading, measured 1.10x parameter
  gate) on 16,384 control blocks held only by `eval/test/`, against a frozen
  teacher copy; failures score worst values in every metric; all artifacts on
  disk and the canonical slots are always scored.
- Reference retrained from final code (seed 13; bit-identical to the red-team
  seed-13 run): 12.136 vs 4.347 accepted tokens for the independent baseline,
  7.074 vs 3.057 N=8 correct tokens (hidden test).
- Longer training (+2.07 accepted, 15x seed range) and an in-cap 4-layer shape
  (+0.54) outrank the reference; all 16 tested shortcuts fail or tie exactly.
  See `HARDENING.md`.

### [2026-11-16] - core_claim_plus (second hardening: training compute)
- Reference O-PTP and independent baseline retrained from the final code for
  the paper's 150k steps (batch 512, warmup + cosine, GPU): hidden test 20.035
  vs 4.347 accepted tokens, 7.830 vs 3.057 N=8 correct tokens.
- In-budget better methods now outrank more training: 4-layer +1.31 accepted
  (rank +0.51), [ar+NN] +0.93 (+0.37) vs 2x training +0.62 / +0.43
  (+0.24 / +0.19); seed range 0.30. The 76.8M-sequence-pass training budget is
  part of the contract. See `HARDENING.md`.
