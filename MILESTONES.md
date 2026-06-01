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
- Completed at milestone `core_claim`. The reproduction implements the paper's
  auxiliary-conditioned O-PTP computation on a scaled synthetic autoregressive
  teacher, evaluates against the independent-prediction ablation, and validates
  that O-PTP obtains longer accepted prefixes at equal parameter scale.
