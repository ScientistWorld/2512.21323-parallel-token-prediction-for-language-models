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
