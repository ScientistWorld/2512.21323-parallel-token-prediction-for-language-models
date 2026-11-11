# Hardening report

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
