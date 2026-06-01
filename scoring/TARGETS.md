# Evaluation Targets

## Primary: synthetic_auxiliary_ablation
Tests whether a parallel sequence proposer obtains longer teacher-verified
prefixes than an independent multi-token proposer.
- **Metric**: `avg_accepted_tokens` (higher is better)
- **How to run**: `bash scripts/evaluate.sh` after method and baseline artifacts
  have been written to `checkpoints/`.

## Constraint: synthetic_auxiliary_ablation
Tests whether gains come from better proposal quality rather than simply using
a larger predictor.
- **Metric**: `relative_param_multiplier` (lower is better)
- **How to run**: `bash scripts/evaluate.sh`.

## Baseline Gap: synthetic_auxiliary_ablation
Tests whether the proposed block generator beats the same-scale independent
multi-token baseline on accepted prefix length.
- **Metric**: `avg_correct_tokens` (higher is better)
- **How to run**: compare `o_ptp` against `independent_prediction` in
  `scoring/scores.json`.

## Additional Setting: synthetic_restricted_compute_n8
Tests the same block-proposal comparison when only the first 8 proposed tokens
are eligible for acceptance.
- **Metric**: `avg_correct_tokens` (higher is better)
- **Constraint**: `relative_param_multiplier` should remain near 1.0.
- **How to run**: `bash scripts/evaluate.sh`.
