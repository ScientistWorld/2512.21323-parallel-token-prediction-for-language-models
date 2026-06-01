# Evaluation Targets

## Primary: codecontests_auxiliary_ablation
Tests whether a parallel sequence proposer obtains longer teacher-verified
prefixes than an independent multi-token proposer.
- **Metric**: `avg_accepted_tokens` (higher is better)
- **How to run**: `bash scripts/evaluate.sh` after method and baseline artifacts
  have been written to `checkpoints/`.

## Constraint: codecontests_auxiliary_ablation
Tests whether gains come from the conditioning mechanism rather than simply
using a larger predictor.
- **Metric**: `relative_param_multiplier` (lower is better)
- **How to run**: `bash scripts/evaluate.sh`.

## Ablation: codecontests_auxiliary_ablation
Tests whether removing informative auxiliary inputs reduces the accepted prefix
length.
- **Metric**: `avg_correct_tokens` (higher is better)
- **How to run**: compare `o_ptp` against `independent_prediction` in
  `scoring/scores.json`.
