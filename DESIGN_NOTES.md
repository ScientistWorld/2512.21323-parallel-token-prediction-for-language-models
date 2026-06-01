# Train/Test Evaluation Design Notes

## Checked
- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`: train calls `eval.train.evaluate` and writes `scoring/scores_train.json`; test calls `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Checked for direct train/test cross-imports and hardcoded single-method dispatch. The evaluators enumerate `*_predictions.npz` artifacts under `checkpoints/` while preserving the standard `o_ptp` and `independent_prediction` aliases.
- Checked that evaluation reads method-produced prediction artifacts and metadata rather than running method code inline.
- Checked that `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran syntax-only checks for both shell scripts and JSON parsing for `scoring/reference.json`.

## Changed
- Added three triangulation metrics to both train and test evaluators:
  - `token_accuracy` to measure all-position prediction quality, not just accepted prefix length.
  - `tail_token_accuracy` to expose methods that only optimize early tokens.
  - `prefix8_success_rate` to add a stricter per-example success metric without using a likely all-zero full-sequence exact match.
- Added a top-level `slice` marker to train/test score outputs so generated artifacts are not ambiguous.
- Updated `scoring/reference.json` so every nonzero metric has `higher_is_better`, `coefficient`, and `description`.
- Increased the parameter multiplier penalty slightly so size-only gains are less attractive.

## Runtime Concerns For Runner Iteration
- I did not run either evaluator or generate scores in this design-only pass. The next runner should confirm that `prefix8_success_rate` is not saturated at all zero or all one on both slices.
- The prompt's paper markdown path was not present on disk, so this pass used the existing workspace experiment descriptions and reference metadata for paper-context checks.
- The split remains a deterministic 50/50 partition of one fixed synthetic artifact batch. This is calibrated and lightweight, but it is still one batch; if future scores show high variance, consider increasing the fixed evaluation artifact size in the method-producing stage rather than changing the evaluator protocol.
