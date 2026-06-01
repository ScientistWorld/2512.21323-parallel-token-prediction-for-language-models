# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-01 06:25] - method_runs
- Implemented the O-PTP computation with causal auxiliary conditioning and arithmetic-style binary embeddings.
- Implemented the independent multi-token baseline and artifact-based evaluator.

### [2026-06-01 06:40] - core_claim
- Verified stored reproduction scores with `python validate.py --compare`.
- O-PTP achieved longer teacher-verified prefixes than the independent baseline at the same parameter scale on the scaled synthetic autoregressive teacher benchmark.

## Stop Justification
- Not stopped; submitting another script-level reproducibility run.
