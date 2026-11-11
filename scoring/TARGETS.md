# Evaluation Targets

All targets are measured by `bash scripts/evaluate_test.sh` (no arguments) and
written to `scoring/scores_test.json`; `bash scripts/evaluate_train.sh` reports
the same quantities on the public development batch.

## Primary: synthetic_auxiliary_ablation
Tests whether a single-call parallel proposer obtains longer teacher-verified
prefixes than an independent multi-token proposer of the same size.
- **Metric**: `avg_accepted_tokens` (higher is better)
- **Also scored**: `avg_correct_tokens`, `token_accuracy`,
  `tail_token_accuracy`, `prefix8_success_rate` (all higher is better)

## Baseline gap: synthetic_auxiliary_ablation
Tests whether the proposed block generator beats the same-scale independent
baseline on accepted prefix length.
- **Metric**: `avg_correct_tokens` of `o_ptp` vs `independent_prediction`

## Additional setting: synthetic_restricted_compute_n8
Tests the same comparison when only the first 8 proposed tokens are eligible.
- **Metric**: `avg_correct_tokens` (higher is better)
- **Also scored**: `avg_accepted_tokens`, `token_accuracy`,
  `tail_token_accuracy`, `prefix8_success_rate`

## Constraint (both experiments): model scale and single call
Gains must come from better proposals, not a larger or sequential predictor.
- **Measured**: the evaluator counts the parameters of the network it builds
  from the artifact (`_constraints.<method>.param_count` and
  `relative_param_multiplier` in the scorecard) and runs it once per block.
- **Enforced**: more than 377,960 parameters (1.10x the 343,600-parameter
  baseline) or any artifact outside the proposer contract scores the run as
  failed, with the worst value of every metric in both experiments.
