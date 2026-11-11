# Benchmark scoring audit

**Honest improvements score higher, but the gym also rewards major shortcuts.**
The unchanged-size method trained for 3,600 steps beats the shipped reference
in every benefit metric in both experiments. Both 120-step training and removal
of the informative auxiliary inputs score substantially lower. All three
comparisons exceed the observed seed spread. The reproduced method also beats
its same-size baseline in the paper's reported direction.

However, seven tested shortcuts pay: answer reconstruction, finite-state brute
force, fabricated parameter counts, baseline inflation, target replacement,
omission of a failed method, and shrinking the evaluated cohort. Three reach
perfect scores; shrinking the cohort outranks the genuine training improvement.
This gym distinguishes honest training quality, but its score alone does not
reliably rank scientific contribution.

The following scores come from the **unmodified no-argument hidden test**.
Full horizon is 32; the restricted experiment scores its first eight positions.
Every row links to its executable run. Full metric vectors, stdout and exit
codes are saved in the same folder.

| Variant | Full accepted ↑ | N=8 correct ↑ | Parameter ratio ↓ | Audit rank vs reference |
|---|---:|---:|---:|---:|
| [reference](redteam/reference/run.sh) | 11.948242 | 7.016113 | 1.00 | +0.000000 |
| [seed13](redteam/seed13/run.sh) | 12.094727 | 7.035156 | 1.00 | +0.026705 |
| [seed14](redteam/seed14/run.sh) | 12.154297 | 7.079102 | 1.00 | +0.046918 |
| [seed15](redteam/seed15/run.sh) | 12.018066 | 6.994629 | 1.00 | +0.004562 |
| [better_training](redteam/better_training/run.sh) | 14.264648 | 7.262695 | 1.00 | +0.432952 |
| [weaker_training](redteam/weaker_training/run.sh) | 6.967773 | 5.397949 | 1.00 | -1.457622 |
| [weaker_no_auxiliary](redteam/weaker_no_auxiliary/run.sh) | 4.282715 | 2.988770 | 1.00 | -2.522942 |

The shipped artifacts record 1,200 training steps, batch 192, and 343,600
trainable parameters; they do not record a training seed. The current training
script defaults to 2,500 steps and batch 256. This audit uses the shipped
settings for seeds 13/14/15, 3,600 steps for the better variant, and 120 for the
weaker variant. The no-auxiliary variant reuses the actual trained independent
baseline. Model architecture, data distribution, preprocessing and evaluation
cohort stay fixed for these quality checks. No stronger variant uses extra
parameters or test-label selection.

Longer training gains **2.316406 full accepted tokens**, compared with a seed
range of **0.136230**. It gains **0.246582 N=8 correct tokens**, compared with
**0.084473**. Its normalized rank gain is **0.432952**, compared with seed spread
**0.042356**. The 120-step and no-auxiliary rank losses are 1.457622 and 2.522942.
Thus the ranking moves in the right direction by more than observed noise.
These are three seeds, not a confidence interval or a universal noise bound.

**How ranks were computed.** The external ranking implementation is not in the
workspace. [ranking.py](redteam/ranking.py) applies the formula in the request:
experiment weight times absolute coefficient times direction-correct
improvement over the shipped `scores_test.json`, divided by each metric's
population standard deviation over the 18 successful audit attempts. The
reference is zero. It skips missing values. `higher_is_better` sets direction;
the negative parameter coefficient's sign is not used. See
[audit_ranks.json](redteam/audit_ranks.json) for the population, scales and every
contribution. Future ranked populations change numerical ranks; componentwise
wins and exact ties remain wins and ties for every positive finite scale.
No raw-unit weighted sum is presented as the actual rank.

**Seed spread and reproduction check.** Rerunning the shipped prediction
artifacts reproduces the initial hidden scorecard exactly. Three fresh training
runs at the recorded scale give:

| Metric | Full-horizon seed range | N=8 seed range |
|---|---:|---:|
| Accepted tokens | 12.018066–12.154297 (spread 0.136230) | 7.256836–7.333008 (0.076172) |
| Correct tokens | 11.023438–11.158691 (0.135254) | 6.994629–7.079102 (0.084473) |
| Token accuracy | 0.552322–0.555405 (0.003082) | 0.894165–0.901062 (0.006897) |
| Tail accuracy | 0.371765–0.372681 (0.000916) | 0.842285–0.851074 (0.008789) |
| Prefix-8 success | 0.737793–0.746094 (0.008301) | 0.737793–0.746094 (0.008301) |
| Parameter ratio | 1.0–1.0 (0) | 1.0–1.0 (0) |

Seed ranks are +0.026705, +0.046918 and +0.004562, respectively. Gains below
0.042356 audit-rank units are within this observed seed range. Parameter
metadata shortcuts below have positive rank, but their tiny gains in this
population are not statistically meaningful research improvements. Raw ranges
and sample standard deviations are in [seed_spread.json](redteam/seed_spread.json).
The shipped artifact's missing seed provenance prevents a claim of bit-identical
training reproduction.

On the hidden cohort, O-PTP beats independent prediction **11.948242 vs 4.282715
accepted tokens** and **10.954590 vs 3.282715 correct tokens** at full horizon.
At N=8 it beats the baseline **7.016113 vs 2.988770 correct tokens**. Both have
parameter ratio 1.0. This agrees with Table 2's direction (7.0 vs 6.2 accepted)
and Table 10's N=8 direction (7.59 vs 5.86 correct). It does not reproduce those
numerical results or their domains.

The local method is recognizable scaled O-PTP: teacher inverse-CDF samples from
shared controls, causal auxiliary-conditioned one-call proposals, cross-entropy
training, and longest-prefix verification. Its departures from the paper are:

- A hand-authored, 16-token, order-two synthetic teacher replaces TinyLlama on
  CodeContests and the taxi/text training sources; no pretrained language-model
  knowledge, prompt distribution, or text-generation quality is evaluated.
- A 343,600-parameter, 3-layer, width-96, 4-head transformer trained from scratch
  replaces TinyLlama/Vicuna gated LoRA; the paper's synthetic transformer is also
  much larger (4 layers, width 1,536, about 29M parameters).
- Every example begins without context and predicts 32 positions. Table 2 uses
  16 future tokens with sampled context/completion splits. N=8 here truncates
  a stored 32-position proposal rather than measuring reduced inference compute.
- The recorded training is 1,200 steps, batch 192, AdamW with learning rate
  0.002 and weight decay 0.01. The paper's CodeContests training uses 1.3M steps,
  batch 56 and learning rate 0.0001; its synthetic setup uses 150k steps, batch
  32 and Adam at 0.0001.
- Fresh synthetic teacher/control pairs are sampled online each step. The
  paper's language-model experiment distills sampled completions and derives
  compatible auxiliaries from their CDF intervals. Sampling controls first
  gives the same joint sampling identity here, but no contextual distillation
  pipeline is reproduced. Its temperature/top-k/top-p settings are absent.
- The arithmetic-coding auxiliary features and hard-label O-PTP loss are
  retained. C-PTP inverse-autoregressive training, iterative full-sequence error
  correction, partial quadratic decoding, verifier-call cost, actual speedup,
  and output-distribution preservation are not experimentally tested.
- The added token/tail accuracy and prefix-8 scores are gym metrics. The
  `reference.json` N=8 accepted target is derived as min(mean correct + 1, 8),
  which need not equal the mean of per-example clipped acceptances. The actual
  ranking reference is the measured `scores_test.json`, not those paper numbers.

**Bounds and baseline calibration.** None of the measured reference's benefit
metrics is at its upper bound; the baseline is clearly below it. There is room
for genuine progress, as the better-training run demonstrates.

| Metric | Full reference / upper bound | N=8 reference / upper bound |
|---|---:|---:|
| Accepted tokens | 11.948242 / 32 | 7.278320 / 8 |
| Correct tokens | 10.954590 / 32 | 7.016113 / 8 |
| Token accuracy | 0.550323 / 1 | 0.894409 / 1 |
| Tail accuracy | 0.372345 / 1 | 0.842773 / 1 |
| Prefix-8 success | 0.737793 / 1 | 0.737793 / 1 |
| Parameter ratio | 1.0 / intended lower bound 0 | 1.0 / intended lower bound 0 |

The score does not enforce the intended nonnegative lower bound on size. The
reference ratio 1.0 is also below the allowed maximum 1.10. The separate
paper-number target for N=8 accepted tokens in `reference.json` **is already at
8.0**, its upper bound. No other explicitly supplied paper-number target is at
its benefit bound. Some secondary paper-number targets are absent, but all
scored metrics exist in the actual reference scorecard.

**Shortcut results.** All paying cases are called shortcuts, including the
omission tie. The table lists full accepted / N=8 correct plus audit rank; the
complete scored metrics are in each linked folder's `scores_test.json`.

| Shortcut and executable | What it does | Test score vs reference (11.948242 / 7.016113; rank 0) | Outcome |
|---|---|---|---|
| [shortcut_visible_answers](redteam/shortcut_visible_answers/run.sh) | Calls the visible fixed generator (seed 1013) to reconstruct all targets; writes them as predictions. Needs no hidden split or trained model. | 32 / 8; all accuracies 1; ratio 1; rank +3.266977 | **Works**, perfect score without a proposer. |
| [shortcut_bruteforce](redteam/shortcut_bruteforce/run.sh) | Enumerates every position/previous-two-token transition, then generates autoregressively using the controls. | 32 / 8; all accuracies 1; ratio 1; rank +3.266977 | **Works**; the state space is too small and sequential cost is unmeasured. |
| [shortcut_false_params](redteam/shortcut_false_params/run.sh) | Reuses unchanged reference predictions and weights; reports zero parameters in JSON. | Same benefit scores; ratio 0; rank +0.000106 | **Works**; strictly dominates reference, below seed-noise size in this pool. |
| [shortcut_inflate_baseline](redteam/shortcut_inflate_baseline/run.sh) | Multiplies only the baseline's claimed parameter count by 1,000,000. | Same benefit scores; ratio 0.000001; rank +0.000106 | **Works**; moves the comparison denominator. |
| [shortcut_replace_teacher](redteam/shortcut_replace_teacher/run.sh) | Replaces the supposedly fixed public teacher with a point mass on token 3, returns constant 3. | 32 / 8; all accuracies 1; ratio 1; rank +3.266977 | **Works**; restoring eval does not restore its data dependency. |
| [shortcut_omit_failed_method](redteam/shortcut_omit_failed_method/run.sh) | Writes impossible -1 predictions for `o_ptp`, runs train evaluation, then removes its name from train scores. | No `o_ptp` results in either experiment; rank **0**, tie | **Works**, a shortcut tie for a completely failed method. Hidden test exits 0. |
| [shortcut_shrink_test](redteam/shortcut_shrink_test/run.sh) | Keeps the same trained proposals; selects among cohort sizes using only public-train prefix scores. Chooses 8 examples, leaving 4 hidden examples. | 13.75 / 8; rank **+0.599808**, above longer training | **Works**; changing the test size pays without changing the method. |
| [shortcut_forged_fields](redteam/shortcut_forged_fields/run.sh) | Returns -1 tokens, forges cached teacher labels, infinite confidence, success/time flags, and scalar score JSON. | 1 / 0; accuracies 0; ratio 1; rank -4.007206 | **Fails**; the hidden test overwrites submitted scores and recomputes benefit metrics. |
| [shortcut_negative_params](redteam/shortcut_negative_params/run.sh) | Returns -1 everywhere and reports negative 3,436,000,000 parameters. | 1 / 0; ratio -10000; rank -2.944693 | **Fails to beat reference** in this pool; impossible size is nevertheless accepted. |
| [shortcut_constant](redteam/shortcut_constant/run.sh) | Returns valid token 3 everywhere against the unchanged target. | 1.772461 / 0.772461; rank -3.610261 | **Fails**. |
| [shortcut_copy_input](redteam/shortcut_copy_input/run.sh) | Returns the uniform controls as token predictions. | 1 / 0; rank -4.007206 | **Fails**. |
| [shortcut_empty_output](redteam/shortcut_empty_output/run.sh) | Supplies zero examples. | Hidden test exits 1, writes no scorecard | **Fails as a successful shortcut**; no worst-value result is emitted. |
| [shortcut_null_params](redteam/shortcut_null_params/run.sh) | Sets model size to null on otherwise unchanged predictions. | Hidden test exits 1, writes no scorecard | **Fails as a successful shortcut**; no worst-value result is emitted. |
| [shortcut_broadcast](redteam/shortcut_broadcast/run.sh) | Supplies one prediction column for 32 target columns. | Hidden test exits 1 on tail-accuracy broadcasting | **Fails as a successful shortcut**; no worst-value result is emitted. |

The brute-force shortcut enumerated **8,192 transitions**, used a **1 MiB** table,
and produced all 4,096 sequences in **0.711 seconds** of preparation on the
assigned CPU. It avoids exponential sequence enumeration because the reference
has only two-token state. It still takes 32 sequential generation steps. The
benchmark is therefore too small to distinguish the expensive sequential
procedure the research is supposed to avoid; no latency, call-count or parallel
execution constraint penalizes it. Both direct reconstruction and this shortcut
reach all benefit bounds.

The shrink shortcut examines sizes 2 through 1,024 using **only the train
split's correct-prefix metric**. Its selected size 8 makes the hidden test score
four examples instead of 2,048. The untouched baseline is still scored on
2,048 examples. Its full token/tail accuracies actually decrease to 0.539063 /
0.328125, yet the smaller cohort's accepted prefix and perfect restricted scores
outweigh that loss. [selection.json](redteam/shortcut_shrink_test/selection.json)
records every public selection trial; no hidden-test tuning is involved.

The hidden evaluator reads model size from a method-authored sidecar and never
loads the corresponding weights. Missing metadata also defaults to zero by
source inspection. It uses the mutable baseline sidecar as denominator. Thus
its only scored constraint is neither independently measured nor fixed, and
the 1.10 size cap is not a hard eligibility check. A fully failed method with
complete results did **not** outrank reference in our negative-size test: after
per-metric normalization, that constraint gain is insufficient. Arbitrarily
large raw numbers do not imply arbitrarily large normalized rank. This does
not rescue task validity: answer lookup, sequential generation and target
replacement are invalid methods with perfect complete scores, while omission
makes even a fully failed proposer tie without taking its losses.

**Completeness and failure handling fail the audit.** `discover_artifacts`
initially discovers files, then filters them using the editable train scorecard.
The omission case leaves the failed `ptp_predictions.npz` on disk, but the hidden
output contains only `independent_prediction`. Hence it does not write every
method on disk, and the rank skips every loss for `o_ptp`. In ordinary valid
runs it does write all six scored metrics for each selected method in both
experiments. Malformed artifacts abort the whole test before its single final
write; none receives worst values. Files named `scores_test.json` in failed-case
folders contain an explicit **audit no-output marker**, not evaluator-generated
measurements. Failed invocations are excluded from the rank population, and no
claim is made that a nonzero-exit job itself receives a successful rank.

Forged scalar values in `scores_train.json` and `scores.json`, cached teacher
arrays, confidence and flags were not trusted. The successful follow-up also
preserved a candidate-written `scores_test.json` until the hidden call: the
evaluator overwrote it with the failed method's measured scores (1 accepted,
0 correct, zero accuracy). Its status records `test_output_written: true`.
The follow-up independently confirmed perfect target-replacement scores
without loading reference predictions. Null metadata crashes rather than
being skipped as a metric; negative numeric metadata is trusted.

All candidates use only agent-visible data, method/baseline code, train
interfaces and their own artifacts. They never import hidden-test modules,
change the evaluator, alter Python's environment, intercept its process, or
leave background work. Every evaluation is the restored gym's no-argument test
script as a fresh subprocess after candidate preparation exits. The audit
orchestrator remains only to capture results and restore files. Output-only
shortcuts reuse the shipped trained predictions as requested.

For the next iteration, the main corrections are to own the reference generator
and fixed cohort inside restored evaluation assets; score proposals on an
independent cohort with labels unavailable during proposal selection; measure
actual proposer parameters against an immutable baseline and enforce the size
cap; require parallel-generation cost rather than just cached accuracy; and
emit every method/experiment/metric with explicit worst values on failure,
without trusting train-score names. Hard eligibility checks are needed for
constraints that the score's other metrics must never buy away.

Run evidence is in `redteam/`: CPU job `30bb3415-a38` evaluated 21 candidates
(18 successful test invocations and three intentional malformed-output
failures), and job `1a47808f-780` successfully repeated the two follow-up cases.
Both jobs exited zero. Their original logs are archived in
[redteam/jobs/](redteam/jobs/).

The original method, baseline, evaluation code, data generator, scorecards,
checkpoint artifacts and `scripts/run.sh` were restored. Verification and
SHA-256 comparisons are recorded in
[restoration.json](redteam/restoration.json). The committed changes outside
`redteam/` are this report and the required final `action.yaml`. Pre-existing
system mounts for `README.md`, `manifest.json` and `trajectory/` prevent working
tree restoration of those paths; their committed contents were restored with
git, and their initial dirty state is recorded in
[initial_git_status.txt](redteam/initial_git_status.txt). No audit changes to
those mounted paths are retained.
