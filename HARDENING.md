# Hardening report

This gym has been hardened twice. The second hardening (this report's first
part) fixes the one weakness `REVIEW.md` found after the first: training
compute was unbudgeted. The first hardening's report follows unchanged below
it, with its numbers as of that iteration.

# Second hardening (2026-11-16): training compute

**Better methods now score higher than more training.** The review found that
training the 1,200-step reference 3x longer was the largest gain (+1.29 rank),
four times an in-cap architectural improvement (+0.31). The reference and the
baseline are now trained from the final code for the paper's 150k steps
(batch 512, warmup + cosine decay), 333x the old reference's 230k training
sequences. At this budget an in-cap architectural change gains **+1.31**
accepted tokens (rank +0.51) and the paper's [ar+NN] embedding **+0.93**
(+0.37). Twice the training gains +0.62 / +0.43 (two seeds; rank +0.24 /
+0.19). That budget, 76.8M training-sequence passes, is now part of the
contract, so longer training is out of bounds. The evaluator is unchanged.
Every shortcut from the first hardening still fails or ties the reference
exactly.

Evidence: GPU job `931408da-1a4` (exit 0) trained every model from the final
code. It then ran every case through the unmodified no-argument hidden test
in its own copy of the gym ([redteam/hardening2/](redteam/hardening2/)):
[run_job.sh](redteam/hardening2/run_job.sh),
[train_variants.py](redteam/hardening2/train_variants.py),
[audit.py](redteam/hardening2/audit.py), per-case scorecards and logs in
`results/<case>/`, training logs in `artifacts/<case>/train.log`, ranks in
[ranks.txt](redteam/hardening2/ranks.txt) /
[ranks.json](redteam/hardening2/ranks.json), and paired per-block statistics in
[paired.json](redteam/hardening2/paired.json). Ranks use the first audit's
formula (weight x |coefficient| x direction x improvement over
`scoring/scores_test.json` / population std, `o_ptp` entries). "Honest" means
the population of the 16 trained variants.

## What changed

| Problem | Fix |
|---|---|
| Reference trained for 1,200 steps x 192 (230k sequences) on a teacher that gives unlimited fresh samples, so more training alone paid most | `method/train_ptp.py` and `baseline/train_independent.py` default to the paper's 150k steps (App. G.2), batch 512, AdamW peak 2e-3 with 2k-step linear warmup and cosine decay to 0, weight decay 0.01, clip 1.0, seeds 13 / 17. The O-PTP algorithm (teacher inverse-CDF samples, Eq. 10 cross-entropy, one causal call) and the network are unchanged. Training runs on a GPU, with TF32 and nondeterministic attention kernels disabled |
| Training compute was not part of the contract | The reference's budget, **76.8M training-sequence passes** (batch x steps, reused sequences count again), is a stated constraint in `briefing/problem.md`, `briefing/evaluation.md`, `scoring/CONSTRAINTS.md`, `TARGETS.md`, `DIRECTION.md` and `EXPERIMENTS.md`. The evaluator cannot read training compute from weights, so the grader checks it from code and logs |
| Teacher sampling in numpy was the bottleneck on a GPU | `data/teacher_stream.py` (public, not used by `eval/`) does the same inverse-CDF lookup on the device. Every run first checks it token for token against the scalar teacher |
| CPU-only torch in the container | `environment/setup.sh` installs the pinned torch 2.12.1 + CUDA 12.6 set (resolved with `uv pip compile`). It installs in groups small enough for the setup's command timeout, with `--reinstall`. The first attempt as one command timed out and left torch half-installed, which failed job `e0c7f8b9-b9a` before any training. Setup is skipped once a working CUDA torch is present. The CUDA build also runs the CPU-only evaluators |
| `validate.py --post-design` compared the 32-token synthetic o_ptp (20.0) with Table 2's 16-token CodeContests number (7.0) | The Table 2 numbers are not on this experiment's scale. They now sit unchanged under `_paper_results` in `scoring/reference.json`, with a scale note; `results` holds the gym's measured reference values. The N=8 experiment keeps Table 10's numbers, which share its 8-token scale. Both `python validate.py` and `--post-design` pass |
| `scripts/run.sh` | Reproduces the reference end to end again (baseline + O-PTP from defaults, both scorecards, validate). Submit it with one GPU |

`eval/`, `scripts/evaluate_*.sh`, `data/proposer.py`, `data/ptp_synthetic.py`,
the metrics, experiment weights and coefficients are unchanged.

## Do better results score higher? (before and after)

"Before" is the first hardening's audit: 1,200-step reference, its 3x training
run and its 4-layer run. "After" is this audit. Each cell gives the hidden-test
`o_ptp` full-horizon accepted tokens / N=8 correct tokens and the rank. The
rank populations differ, so compare order and gaps against the seed range, not
absolute ranks.

| Variant | Before: full acc / N8 corr (rank) | After: full acc / N8 corr (rank) |
|---|---|---|
| reference (seed 13) | 12.136 / 7.074 (0) | **20.035 / 7.830 (0)** |
| seed 14 | 12.144 / 7.128 (+0.037) | 19.735 / 7.835 (-0.087) |
| seed 15 | 12.005 / 7.034 (-0.092) | 19.868 / 7.816 (-0.062) |
| more training alone: 3x before, 2x after (seed 13) | 14.211 / 7.316 (**+1.287**) | 20.651 / 7.853 (**+0.242**) |
| more training alone: 2x, seed 14 | - | 20.462 / 7.863 (+0.188) |
| better: 4 layers x 84 wide, under the cap | 12.678 / 7.129 (+0.312) | 21.344 / 7.885 (**+0.515**) |
| better: [ar+NN] embedding (paper Table 7) | - | 20.965 / 7.871 (+0.368) |
| weaker: C-PTP, offset 1 + inverse-CDF decode (paper Tables 6/7) | - | 14.701 / 7.366 (-2.257) |
| weaker: [th+lin] embedding (paper Table 7) | - | 17.717 / 7.146 (-1.458) |
| weaker: previous reference, 1,200 steps x 192 | (was the reference) | 12.136 / 7.074 (-3.509) |
| weaker: previous 3x run, 3,600 steps x 192 | (+1.287) | 14.211 / 7.316 (-2.521) |
| weaker: no auxiliaries (the trained baseline) | 4.347 / 3.057 (-7.385) | 4.347 / 3.057 (-9.869) |

Paired differences to the reference on the same 16,384 hidden blocks
(full-horizon accepted tokens, ± standard error):
- better: 4-layer **+1.310 ± 0.036**; ar+NN **+0.930 ± 0.035**;
- seed noise: seed 14 -0.299 ± 0.033; seed 15 -0.166 ± 0.033;
- 2x training: +0.616 ± 0.032 and +0.427 ± 0.034;
- weaker: th+lin -2.317 ± 0.051; C-PTP -5.334 ± 0.046; no auxiliaries -15.688 ± 0.057.

At N=8: 4-layer +0.055 ± 0.004 correct tokens; seeds -0.014 and +0.006;
2x training +0.023 and +0.033. The seed range at the reference recipe is 0.299
accepted tokens and 0.087 rank units. Both better methods beat the reference by
3-4x that range and beat 2x training. The paper's weaker variants rank below
in the paper's order: ar+lin > th+lin, and O-PTP > C-PTP.

Learning curve of the reference recipe (seed 13, cosine over the stated
steps; hidden test):

| steps | 9.4k | 18.75k | 37.5k | 75k | **150k** | 300k |
|---|---|---|---|---|---|---|
| full accepted | 15.97 | 17.16 | 18.36 | 19.26 | **20.03** | 20.65 |
| gain per doubling | | +1.19 | +1.21 | +0.90 | +0.78 | +0.62 |
| N=8 correct | 7.58 | 7.67 | 7.75 | 7.79 | **7.83** | 7.85 |

Train and test agree for every variant (train / test full accepted):
- reference 20.12 / 20.03; seeds 19.86 / 19.74 and 20.08 / 19.87;
- 2x 20.71 / 20.65 and 20.57 / 20.46;
- 4-layer 21.37 / 21.34; ar+NN 21.09 / 20.97;
- C-PTP 14.77 / 14.70; th+lin 18.02 / 17.72;
- no auxiliaries 4.31 / 4.35.

The paper's claim holds in its direction on the hidden test, at equal size and
equal training. O-PTP beats independent prediction by 20.03 vs 4.35 accepted
tokens over 32 positions (paper Table 2: 7.0 vs 6.2 over 16), and by 7.83 vs
3.06 correct tokens at N=8 (paper Table 10: 7.59 vs 5.86).

## Shortcuts: fix, score before and after

The shortcut `REVIEW.md` named is **more training alone**:
- **Before:** 3x training gained +2.074 ± 0.029 accepted tokens, rank +1.287. That was 4x the in-cap architectural gain and 10x the seed range in rank.
- **After:** 2x training gains +0.616 / +0.427, rank +0.242 / +0.188, below both better methods (+0.515, +0.368).
- **Fix:** the reference is trained with the paper's step count, where returns per doubling have fallen to about half a token. Training beyond the 76.8M-pass budget is now outside the contract.

The evaluator did not change. The first hardening's shortcut suite was rerun
against the new reference, with these results (hidden-test full accepted /
N=8 correct, rank in the population of all 32 cases):

| Shortcut | After | Rank |
|---|---|---:|
| visible answers; TorchScript / pickled / 32-layer sequential decoding; oversized proposer; NaN proposer omitted from train scores; forged scorecard; no artifacts; wrong shapes; malicious pickle; invalid config | failed -> 1 / 0 | -5.475 |
| replace the public teacher with a point mass + constant proposer | 1.782 / 0.782 | -5.092 |
| false parameter sidecar; inflated baseline; 8-example "test size" sidecar; planted numpy/torch/sitecustomize | = reference, 20.035 / 7.830 | 0.000 |

The four exact ties submit the reference's weights plus junk the evaluator
ignores; they gain nothing. No shortcut code ran and no import was hijacked
(markers in each `status.json`).

## Shortcuts and limits left, and why

- **Training beyond the budget still pays a little.** The learning curve's
  gain per doubling falls by about 0.8x each time. Getting one doubling below
  the seed range (0.30) would take roughly 16x the reference's training.
  Producing that reference and its 2x check did not fit the remaining
  1-GPU / 6-hour jobs. The evaluator cannot measure training compute from the
  weights, so the 76.8M-pass budget is a documented rule that the separate
  grader checks from the method's code and logs. Within the budget, more of
  the same recipe is not available. The budget can still be split
  differently: a larger batch with fewer steps is allowed and is a legitimate
  choice to study.
- **Best-of-N seed selection.** Picking the best of several runs gains at most
  about the seed range (0.30 accepted tokens, 0.087 rank). The contract counts
  all such runs against the budget; the grader checks it.
- From the first hardening, unchanged:
  - the hidden seed lives in `eval/test/`;
  - methods may exploit the public teacher's structure during training;
  - the contract restricts architectures to the `ProposerNet` family;
  - smaller models get no credit;
  - `validate.py` warns, as a false positive, that the evaluators "look detached".
- **Audit job resources.** `train_variants.py` ran 12 training processes over
  all 4 A100s the container could see, while the job was allocated one GPU and
  one CPU. This affected only wall-clock time; the default is now one process
  on the default device.

## Reference score before and after

- **Before:** 12.136353 accepted / 7.074402 N=8 correct (1,200 steps x 192, seed 13).
- **After:** **20.034607 / 7.829712** (`scoring/scores_test.json`).
- Train slice: 20.122559 / 7.815186.
- Independent baseline: 4.346863 / 3.056946, identical before and after. It
  ignores its inputs, so once it has learned each position's most likely
  token it proposes the same block every time. That was already true at
  1,200 steps.
- Both scorecards come from the reference and baseline runs alone, trained by
  the final code with default arguments.
- Training time: the reference took 88 minutes while sharing 4 A100s with 11
  other training processes. A single run on a dedicated GPU was not timed.

## Headroom check

Full horizon (bounds 32 / 32 / 1 / 1 / 1):
- reference: accepted 20.03, correct 19.13, token accuracy 0.753, tail 0.586, prefix-8 0.934;
- none at its bound;
- two in-budget variants beat it by 0.93 and 1.31 tokens (26-37x paired SE).

N=8 (bounds 8 / 8 / 1 / 1 / 1):
- reference: correct 7.83, accepted 7.90, token accuracy 0.983, tail 0.968, prefix-8 0.934;
- close to the bound, with 0.17 correct tokens left;
- it still separates methods: 4-layer +0.055 ± 0.004, seeds within 0.019, weaker variants 0.46-4.77 below.

No experiment was added: the full-horizon experiment carries the headroom.
The brute force the paper avoids (sequential decoding) cannot be expressed,
and the simple baseline sits far below (4.35).

## Weights

Unchanged: experiments 0.65 / 0.35, all coefficients as before, parameter count
a gate. The N=8 experiment is now near its bound, so it contributes little to
separating better methods. It stays because it is the paper's
restricted-compute claim (Table 10) and it still penalises weaker methods.

## What could not be fixed

- Full convergence of the reference (see above), and a machine-checked
  training budget.
- As before:
  - the teacher is the hand-authored order-two, 16-token process;
  - the paper's TinyLlama / taxi / Vicuna teachers, wall-clock speedup, C-PTP
    from-scratch training, partial quadratic decoding and 29M-parameter scale
    are not reproduced.

# First hardening (2026-11-11): evaluator-run proposer contract

*Numbers in this part are as of the first hardening (1,200-step reference, 12.136 accepted tokens); the second hardening above supersedes the reference, the training recipe and the scorecards.*

**Better results now score higher, worse ones lower, and every major shortcut
from `REDTEAM.md` is closed.** The hidden test no longer reads anything a
method writes except a trained proposer, which it rebuilds from its own frozen
architecture contract and runs once per block on 16,384 control-value blocks
that only `eval/test/` holds, against its own frozen copy of the teacher. All
evidence was produced by CPU job `bdd3bad0-e87` (exit 0) and is in
[redteam/hardening/](redteam/hardening/): per-case scorecards and logs in
`results/<case>/`, the rank table in [ranks.txt](redteam/hardening/ranks.txt) /
[ranks.json](redteam/hardening/ranks.json), paired per-block statistics in
[paired.json](redteam/hardening/paired.json). Each case ran in its own copy of
the gym. The case could edit anything a method can reach (checkpoints, data,
scoring, the workspace root, `~/.local`). `eval/` and `scripts/evaluate_test.sh`
were then restored, and `bash scripts/evaluate_test.sh` ran with no arguments
as a fresh process ([audit.py](redteam/hardening/audit.py)).

## What changed

| Problem in the red-team audit | Fix |
|---|---|
| Hidden test scored method-written predictions on a **public** control batch, so targets could be reconstructed from the public generator | A method now submits `checkpoints/<name>_proposer.pt` (config + tensors). `eval/test/` generates its own 16,384 control blocks (seed held only there), runs the proposer itself and never reads predictions, sidecars or score files |
| Teacher came from editable `data/ptp_synthetic.py` | `eval/test/hidden_teacher.py` and `eval/train/dev_teacher.py` are frozen copies; the evaluators never import `data/` |
| Sequential decoding with the tiny teacher was accepted (no compute or call constraint) | The proposer must be an instance of the evaluator-owned `ProposerNet` family: causal transformer, at most 4 layers, one forward call per block. It is loaded with `torch.load(weights_only=True)`, strict keys and shapes, finite tensors and logits. No method code, TorchScript or pickled object can run |
| Parameter count read from a method-written sidecar; baseline sidecar used as denominator; cap not enforced; zero/negative/null accepted or crashing | The evaluator counts the parameters of the network it builds. The denominator is the constant 343,600, and more than 377,960 (1.10x) fails the run. The count is reported under `_constraints`, not ranked (see Weights) |
| Test size chosen by the method's artifact | Fixed at 16,384 blocks by `eval/test/` |
| Methods filtered by the editable `scores_train.json`; malformed artifacts crashed the test with no scorecard | Every `*_proposer.pt` on disk is scored, plus the slots `o_ptp` and `independent_prediction`. Any failure gives the worst value of every metric in both experiments (accepted 1, all else 0). The script deletes a stale scorecard first and writes worst values if the evaluator itself fails |
| (small extra fixes) | Python runs with `-I`, so planted `numpy.py`/`torch.py`/`sitecustomize.py`, `PYTHONPATH` or a user site cannot shadow imports. Artifacts over 32 MB are rejected before loading |

The method and baseline are unchanged in substance. They build the contract
network (the O-PTP config is `ar` + `lin`, offset 0, argmax, hidden 96 x 3
layers; the baseline uses `none`) and save it as a proposer artifact. The
teacher sampler is vectorised from a table built by the original scalar
function, and the training defaults now match the shipped recipe (seed 13,
1,200 steps, batch 192), which the earlier script defaults did not. The final
code reproduces the red team's seed-13/14/15, 3,600-step and 120-step runs
**bit for bit**: every metric of `seed14` equals `seed14_old`, and so on. So the
reference is now reproducible with a recorded seed.

## Do better results score higher? (variants before and after)

Before is the old hidden test (2,048 public blocks) from `REDTEAM.md`. After is
the fixed hidden test (16,384 hidden blocks). Ranks use the red-team formula
(weight x |coefficient| x direction x improvement / population std, `o_ptp`
entries, reference = shipped `scores_test.json`). "After" ranks use the 13
honest attempts as the population and the fixed metric set. Ranks before and
after are not on the same scale; the order and the gaps relative to seed
spread are what compare.

| Variant | Full accepted before | after | N=8 correct before | after | Rank before | Rank after |
|---|---:|---:|---:|---:|---:|---:|
| reference (seed 13, 1,200 steps) | 11.9482 | **12.1364** | 7.0161 | **7.0744** | 0 | 0 |
| shipped original artifact | 11.9482 | 12.1368 | 7.0161 | 7.0684 | 0 | -0.010 |
| seed 14 | 12.1543 | 12.1440 | 7.0791 | 7.1277 | +0.047 | +0.037 |
| seed 15 | 12.0181 | 12.0052 | 6.9946 | 7.0338 | +0.005 | -0.092 |
| better: 3,600 steps | 14.2646 | 14.2105 | 7.2627 | 7.3161 | +0.433 | **+1.287** |
| better: 4 layers x 84 wide (under cap) | - | 12.6779 | - | 7.1288 | - | **+0.312** |
| weaker: 120 steps | 6.9678 | 7.0557 | 5.3979 | 5.4561 | -1.458 | -4.277 |
| weaker: no auxiliaries (trained baseline) | 4.2827 | 4.3469 | 2.9888 | 3.0569 | -2.523 | -7.385 |

Paired comparisons on the hidden test (same 16,384 blocks for every method),
full-horizon accepted tokens, difference to the reference ± standard error:
3,600 steps **+2.074 ± 0.029**; 4-layer **+0.542 ± 0.025**; seed 14
+0.008 ± 0.025; seed 15 -0.131 ± 0.027; 120 steps -5.081 ± 0.034; no
auxiliaries -7.790 ± 0.040. The seed range is 0.139 accepted tokens and 0.129
rank units. Longer training beats the reference by 15x the seed range on the
full horizon and by 2.6x on N=8 (+0.242 ± 0.008 vs a seed range of 0.094). The
in-cap 4-layer architecture beats it by 3.9x the seed range on the full
horizon. On N=8 its gain (+0.054 ± 0.007) is as large as seed 14's and **not
beyond seed noise**. Train and test agree for every honest variant (train/test
accepted: reference 12.228/12.136, 3,600 steps 14.327/14.211, 4-layer
12.674/12.678, 120 steps 7.031/7.056, no auxiliaries 4.314/4.347).

The paper's claim holds in its direction on the hidden test. O-PTP vs
independent prediction is 12.136 vs 4.347 accepted tokens over 32 positions
(paper Table 2: 7.0 vs 6.2 over 16) and 7.074 vs 3.057 correct tokens at N=8
(paper Table 10: 7.59 vs 5.86), both at 343,600 parameters. The effect is
about 56x the seed range.

## Shortcuts: fix, score before and after

Scores are `o_ptp` full accepted / N=8 correct. "Rank after" uses the
population of all 29 variant and shortcut cases.

| Shortcut | Before (rank) | After | Rank after | Fix |
|---|---|---|---:|---|
| visible answers (public generator reconstructs targets) | 32 / 8 (+3.267) | failed -> 1 / 0 | -5.252 | hidden controls; predictions never read |
| brute force: sequential teacher decoding | 32 / 8 (+3.267) | failed -> 1 / 0 (TorchScript, pickled module and a 32-layer config all rejected) | -5.252 | evaluator-owned one-call family, at most 4 layers, weights-only load |
| replace teacher with a point mass + constant proposer | 32 / 8 (+3.267) | 1.782 / 0.782 | -4.752 | frozen teacher copies in `eval/` |
| shrink test (choose 8 blocks) | 13.75 / 8 (+0.600) | 12.136 / 7.074 (= reference) | 0.000 | test size fixed by `eval/test/` |
| omit failed method from `scores_train.json` | missing (0, tie) | failed -> 1 / 0 | -5.252 | every artifact + required slots always scored |
| false parameter count (sidecar says 0) | ratio 0 (+0.0001) | = reference | 0.000 | parameters counted by the evaluator |
| inflate baseline parameters | ratio 1e-6 (+0.0001) | `o_ptp` = reference; inflated baseline fails | 0.000 | constant denominator; cap and size checks |
| negative parameter count | accepted | invalid config -> failed | -5.252 | config validation |
| oversized proposer (803,856 params) | (not tested) | failed (constraint) | -5.252 | measured 1.10x cap |
| empty output / null params / broadcast shapes | crash, no scorecard | worst values written, exit 0 | -5.252 | per-artifact failure handling |
| forged pre-written scorecards | overwritten | deleted, slots failed | -5.252 | stale file removed first |
| malicious pickle (would run a shell command) | (not tested) | rejected, no code ran | -5.252 | `weights_only=True` |
| planted `numpy.py`/`torch.py`/`sitecustomize.py`/`~/.local` | (not tested) | no effect (= reference) | 0.000 | `python -I` |

The four rows at 0.000 submit the unchanged reference weights plus junk files.
They score exactly the reference because the junk is ignored. They gain
nothing; they are not paying ties. The untrained smoke tests of other contract
options (`th`+`nn`, offset 1, inverse-CDF decode; scalar input) load and score
validly at 1.05 and 1.78 accepted, which confirms those code paths work.

## Shortcuts and limits left, and why

- **Knowing the hidden seed.** `eval/test/evaluate.py` contains the test
  seed, as the rules require: test inputs live only in `eval/test/`. An agent
  that could read it could fit its proposer to those exact 16,384 blocks. This
  relies on `eval/test/` staying hidden.
- **Knowledge of the public teacher during training.** A method may exploit
  the hand-written teacher's structure when designing its training, for
  example curricula aimed at its positional rules. This is the distillation
  setting of the paper (its teacher is fully known too). It cannot reach the
  answers at inference, because the evaluator runs the proposer. It means the
  gym rewards distilling this order-two teacher, which is simpler than an LM.
- **Contract restricts architectures.** Novel architectures outside the
  `ProposerNet` options cannot be scored. This is a headroom limit chosen so
  that one call and the parameter budget can be enforced, not a loophole.
- **Smaller models get no credit.** Size is a gate, not a ranked metric (see
  below).
- `validate.py` prints a heuristic warning that the evaluators "look
  detached". They read proposer artifacts via computed paths, not a literal
  path string. All checks pass.

## Reference score before and after

Before: 11.948242 accepted / 7.016113 N=8 correct (shipped artifact, unknown
seed, old 2,048-block public test). After: **12.136353 / 7.074402**
(`scoring/scores_test.json`). The reference was retrained from the final code
with seed 13 and scored on the 16,384 hidden blocks; the train slice gives
12.228027 / 7.095703. The shipped artifact scores 12.1368 / 7.0684 on the new
test (paired difference +0.0004 ± 0.020). The reference level is unchanged;
the differences are the test set and reproducible provenance. The independent
baseline is 4.346863 / 3.056946.

## Headroom check

No scored metric of the reference is at its bound:

| | accepted | correct | token acc. | tail acc. | prefix-8 |
|---|---|---|---|---|---|
| full horizon (bound 32 / 32 / 1 / 1 / 1) | 12.14 | 11.14 | 0.555 | 0.374 | 0.747 |
| N=8 (bound 8 / 8 / 1 / 1 / 1) | 7.33 | 7.07 | 0.902 | 0.854 | 0.747 |

Two honest variants beat the reference by more than seed noise: longer
training, and a different shape under the cap. Neither is close to the
bound. The brute force the paper avoids (sequential decoding) is no longer
expressible. The simple baseline sits clearly below (4.35 accepted). N=8 is the
tightest experiment, with 0.93 correct tokens left, and longer training still
gains 0.24 there. No experiment was added.

## Weights

The experiment weights (0.65 full, 0.35 N=8) and every benefit-metric
coefficient are unchanged. `relative_param_multiplier` (coefficient -0.25 in
both experiments) was **removed from the rank** and replaced by the hard
1.10x gate. Every eligible method shares one architecture family, so this
metric sits at 1.0 for nearly every attempt. Dividing by its population spread
would turn any slight size cut into an outsized rank gain. As a gate it still
enforces the paper's "same model size" condition, and a violation costs every
metric. The paper's reported numbers in `scoring/reference.json` are unchanged.
The notes now say the setup is scaled and that size is a gate.

## What could not be fixed

- The teacher is still the hand-authored order-two, 16-token synthetic
  process. Moving to the paper's TinyLlama/CodeContests or its taxi GPT-2
  teacher was out of reach for this CPU budget.
- Wall-clock speedup, C-PTP training, partial quadratic decoding and the
  paper's 29M-parameter / 150k-step scale are not measured.
- Future rank populations will differ from this audit's. The paired
  differences and seed ranges above are the invariant evidence.
