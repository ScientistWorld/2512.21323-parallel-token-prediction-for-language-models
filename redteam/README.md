# Scoring audit runs

`bash redteam/run_all.sh` bundles all runs into one CPU job. Each candidate has
its own `run.sh` and preparation code. The runner restores the base data,
prediction artifacts and score files before and after each case, runs the
candidate as a completed subprocess, runs the public train evaluator, and
invokes **`bash scripts/evaluate_test.sh` with no arguments** as a fresh process.
It never changes `eval/`, test scripts, Python imports, or the container.

Only visible method/data/train interfaces are used in candidate code. Trained
reference outputs are reused for output-only attacks as authorized by the audit.
The answer reconstruction, finite-state brute force and target replacement
shortcuts also work from scratch with no reference training.

`reference/scores_initial.json` is the shipped hidden scorecard;
`reference/scores_test.json` is its rerun. Per-case `scores_test.json` files are
copied only after the no-argument hidden test, with return codes in `status.json`.
A failed test with no output gets an explicit audit marker, never fake metrics.
Prediction archives, trained weights and restoration backups remain ignored.

The three seed reruns use the shipped training setting (1,200 steps, batch 192),
not the later script default. Better training uses 3,600 steps at unchanged
model size; weak training uses 120. The key-idea-removed variant is the existing
trained no-auxiliary baseline. All use the same 4,096-example evaluation batch.

`audit_ranks.json` applies the rank formula in the request using a declared
population: all successful audit candidates. Directions use
`higher_is_better`, coefficients use absolute magnitude, experiment weights are
preserved, missing values are skipped, and per-metric improvement is divided by
population standard deviation. Future attempts change that standard deviation;
raw metric vectors and componentwise dominance are the invariant evidence.

The two completed jobs' original logs are archived under `jobs/` in this audit
directory. `python3 redteam/verify.py` performs lightweight evidence, syntax,
and restoration checks and writes `restoration.json`. It does not run training
or model inference. System-mounted paths that were already dirty when the audit
started are documented separately in `restoration_commands.json`.
