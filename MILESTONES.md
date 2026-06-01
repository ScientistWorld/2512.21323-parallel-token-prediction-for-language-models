# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:55] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8 and N=16.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.
