# Overview

- **Paper ID:** 2512.21323
- **Title:** Parallel Token Prediction for Language Models
- **Domain:** Language model inference / speculative decoding
- **TL;DR:** PTP feeds sampling randomness into a causal model so several
  dependent future tokens can be proposed in one model call.

## Short Summary

Autoregressive decoding is slow because each token normally requires a new
forward pass. Parallel Token Prediction moves the random variables used for
sampling into the model input, letting future tokens be predicted jointly while
remaining verifiable against an autoregressive teacher. The paper reports that
O-PTP improves accepted tokens and speedup over independent and speculative
decoding baselines.

## Key Results

- Table 2: O-PTP obtains 7.0 accepted tokens versus 6.2 for independent
  prediction on CodeContests/TinyLlama.
- Table 3: O-PTP reports 2.40x average speedup and 4.21 accepted tokens on
  SpecBench with Vicuna teacher verification.

---

# Problem Definition

Autoregressive language and sequence models generate one token per model call.
The problem is to increase the number of future tokens that can be safely
obtained per verification step while preserving the distribution of a reference
autoregressive generator.

The reference generator (the "teacher", `data/ptp_synthetic.py`) samples each
token by inverse-CDF lookup of one uniform random control value in its
conditional distribution given the previous tokens. A proposer receives the
control values for a block of 32 future positions (no context) and must output
all 32 tokens in a single forward call of a model, before any of the teacher's
tokens are known. The verifier then samples the same positions from the teacher
with the same control values. A proposed block is useful only up to the first
token that disagrees with the verifier.

A successful method maximizes the average length of the accepted prefix under
verifier comparison. It must do so at fixed model scale, fixed training
compute and within one model call per block: the proposer is a model from a
fixed family of causal transformers (see `briefing/evaluation.md`), at most
1.10x the parameter count of a same-scale multi-token baseline, trained with
at most 76.8 million training-sequence passes (the reference's own budget),
and the evaluator itself runs it on control values that only the evaluator
holds. Within those limits, methods are free in how they train that model.

---

# Paper's Method

The paper proposes Parallel Token Prediction (PTP). Instead of drawing a token
from a predicted categorical distribution after a model call, PTP exposes the
uniform random variable used for inverse-CDF sampling as an input to the model.
Given a prefix and auxiliary variables `u_i ... u_k`, future tokens become
deterministic functions of those variables and can be predicted jointly.

The reproduced core path is One-Hot PTP distillation. A teacher autoregressive
sampler produces training sequences from uniform auxiliaries. The student is a
causal transformer over structured 32-dimensional auxiliary embeddings, using
cross-entropy to predict the teacher's sampled token at every future position.
Because the causal mask lets position `k` attend only to `u_i ... u_k`, the
network implements the paper's O-PTP conditioning pattern.

The baseline is the paper's independent-prediction ablation: the same transformer
shape predicts all future positions with uninformative auxiliary inputs. It can
learn marginal position-wise behavior but cannot coordinate later tokens with
earlier sampled outcomes.

Both systems are instances of the benchmark's proposer contract
(`data/proposer.py`): O-PTP uses `aux_embedding="ar"`, `aux_head="lin"`,
`aux_offset=0`, `decode="argmax"`; the baseline uses `aux_embedding="none"`.
Both are hidden 96, 3 layers, 4 heads (343,600 parameters), trained with AdamW
(peak lr 2e-3 after a 2,000-step linear warmup, cosine decay to 0, weight
decay 0.01, clip 1.0) for the paper's 150,000 steps of 512 fresh teacher
samples each (seeds 13 and 17), which is the gym's whole training budget. A GPU
is used when available (`data/teacher_stream.py` samples the teacher on the
device); training the same recipe for 9.4k to 300k steps gives 16.0, 17.2,
18.4, 19.3, 20.0 and 20.7 accepted tokens on the hidden test, so the reference
sits where each doubling of training adds well under one token.

At evaluation the evaluator runs each proposer once per block on its own
control values; the teacher verifier samples the reference sequence with the
same auxiliaries, and the score is the longest matching prefix plus the free
verifier token, matching the paper's sequential speculative decoding metric.

Deviation from the full paper scale: this workspace uses a small synthetic
autoregressive teacher (order two, 16 tokens) rather than TinyLlama/CodeContests,
the taxi GPT-2 teacher or Vicuna/SpecBench, a 32-token horizon instead of 16 or
100, and a 343,600-parameter student trained from scratch for 150k steps of
batch 512 instead of 29M parameters for 150k steps of batch 32 (or LoRA
finetuning for 1.3M steps). The algorithmic steps are preserved; model size
and data source are scaled down, while the training length follows the paper's
step count.

---

# Evaluation

## What a method submits

One file per method: `checkpoints/<method>_proposer.pt`, written with
`data.proposer.save_proposer(path, model)` for a `data.proposer.ProposerNet`.
It holds a small config and a tensor state dict, nothing else. The evaluators
rebuild the network from their own frozen copy of that contract, load the
tensors with `torch.load(weights_only=True)`, count the parameters themselves,
and call the model once per block of 32 control values.

Fixed by the evaluator: vocabulary 16, horizon 32, causal self-attention,
learned positional embedding, pre-norm GELU encoder layers, one forward call.
Chosen in the config (`data.proposer.CONFIG_SPACE`):

| key | allowed |
|---|---|
| `hidden_size` | int 8-512, divisible by `heads` |
| `layers` | int 1-4 |
| `heads` | int 1-16 |
| `ff_mult` | int 1-8 (feed-forward width = hidden x ff_mult) |
| `aux_embedding` | how the control value enters position k: `ar` (its 32 binary digits), `th` (32 thresholds 1{u <= j/32}), `fl` (the scalar), `none` (ignored) |
| `aux_head` | `lin` (affine) or `nn` (2-layer GELU MLP) map of those features |
| `aux_offset` | `0`: position k sees controls 1..k; `1`: controls 1..k-1 |
| `decode` | `argmax` of the logits, or `pick` (inverse-CDF of the softmax at the position's control value) |

**Hard gates.** A run is failed, and receives the worst value of every scored
metric in both experiments (accepted 1, everything else 0), if its artifact is
missing, cannot be loaded as tensors only, has a config outside the table,
tensors that do not match the config or are not finite, non-finite logits, or
more than 377,960 parameters (1.10x the 343,600-parameter baseline). The slots
`o_ptp` and `independent_prediction` are always scored (failed if absent).
Sidecar JSON files, prediction arrays and score files written by a method are
never read.

**Training budget.** Comparisons are at equal training compute. The submitted
proposer, together with any helper network its training depends on, may be
trained with at most **76,800,000 training-sequence passes**: summed over all
optimisation steps, the number of 32-position sequences processed (batch size
x steps; a sequence reused in a later epoch counts again). This is exactly the
reference's budget (150,000 steps of 512 sequences). Drawing control values
and querying the reference generator to build training data is not counted.
Picking the best of several finished training runs by their scores spends
more compute; if you do it, the runs count together. The evaluator cannot
measure training compute from the weights, so this rule is checked from the
method's code and logs; record steps and batch sizes in your training output.

## Metrics

Per block the evaluator compares the decoded proposal with the teacher's tokens
under the same control values.

- `avg_correct_tokens`: mean length of the longest matching prefix.
- `avg_accepted_tokens`: mean of `min(correct + 1, horizon)`; the verifier
  supplies one token at the first rejection (sequential correction).
- `token_accuracy`: fraction of all positions that match.
- `tail_token_accuracy`: match rate over the second half of the horizon.
- `prefix8_success_rate`: fraction of blocks whose first 8 tokens all match.

`synthetic_auxiliary_ablation` scores the full 32-token horizon;
`synthetic_restricted_compute_n8` scores the first 8 positions of the same
proposals (the model is causal, so this equals an 8-token call). The parameter
count and its ratio to the baseline are reported under `_constraints` in each
scorecard; they gate eligibility and are not ranked.

## Slices

- `bash scripts/evaluate_train.sh` (and `scripts/evaluate.sh`, which writes
  `scoring/scores.json`): the public development batch, 4,096 blocks, seed
  1013 (`data.ptp_synthetic.fixed_eval_batch`) -> `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: no arguments; a held-out batch of 16,384
  blocks known only to the evaluator -> `scoring/scores_test.json`.

Both slices draw control values from the same uniform distribution and use the
same code path, so an honest method scores alike on both; only the specific
control values differ.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:00] - none
- Read the paper markdown and extracted the core PTP distillation and teacher
  verification procedure.
- Implemented the scaled O-PTP and independent-prediction pipelines, but no
  compute job has completed yet, so no milestone above `none` is claimed.

### [2026-06-01 06:10] - none
- Audited the existing workspace. The implementation contains the paper's PTP
  computational steps at small scale, and eval remains separated from method.
- No job artifacts or scores were present, so the milestone remains `none`
  pending a real run.

### [2026-06-01 06:35] - core_claim
- Completed the CPU reproduction job for the scaled synthetic benchmark.
- O-PTP produced 12.189 average accepted tokens and 11.195 average correct
  tokens, compared with 4.314 and 3.314 for the independent predictor.
- Both methods use the same parameter multiplier, so the result supports the
  paper's core claim that meaningful auxiliary conditioning improves
  teacher-verified multi-token prediction without increasing model size.
- `python validate.py --compare` passes, including method/eval separation and
  score/reference schema checks.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:25] - method_runs
- Implemented the O-PTP computation with causal auxiliary conditioning and arithmetic-style binary embeddings.
- Implemented the independent multi-token baseline and artifact-based evaluator.

### [2026-06-01 06:40] - core_claim
- Verified stored reproduction scores with `python validate.py --compare`.
- O-PTP achieved longer teacher-verified prefixes than the independent baseline at the same parameter scale on the scaled synthetic autoregressive teacher benchmark.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8 and N=16.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal lengths N=8 and N=16, supporting
  the paper's claim across additional compute settings.
- `python validate.py --compare` passes with the regenerated `scores.json`.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

### [2026-06-01 07:03] - core_claim_plus
- Completed a packaging audit of the scorecard and found the main
  `scoring/scores.json` omitted secondary metrics declared in
  `scoring/reference.json`.
- Updated the evaluator and score artifact so the official scorecard includes
  token accuracy, tail-token accuracy, prefix-8 success rate, and the
  parameter-count constraint.
- Submitting a CPU test job to rerun `scripts/evaluate.sh` inside the managed
  container because the login-node Python environment lacks `numpy`.

### [2026-06-01 07:08] - core_claim_plus
- Final verification confirmed the scaled O-PTP reproduction supports the core
  accepted-prefix claim and the restricted N=8 setting.
- Tightened method-agnostic documentation and made the main evaluator discover
  arbitrary compatible prediction artifacts under `checkpoints/`.
- Documented derived reference metrics and completed `VERIFICATION.md`.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

### [2026-06-01 07:03] - core_claim_plus
- Completed a packaging audit of the scorecard and found the main
  `scoring/scores.json` omitted secondary metrics declared in
  `scoring/reference.json`.
- Updated the evaluator and score artifact so the official scorecard includes
  token accuracy, tail-token accuracy, prefix-8 success rate, and the
  parameter-count constraint.
- Submitting a CPU test job to rerun `scripts/evaluate.sh` inside the managed
  container because the login-node Python environment lacks `numpy`.

### [2026-06-01 07:08] - core_claim_plus
- Final verification confirmed the scaled O-PTP reproduction supports the core
  accepted-prefix claim and the restricted N=8 setting.
- Tightened method-agnostic documentation and made the main evaluator discover
  arbitrary compatible prediction artifacts under `checkpoints/`.
- Documented derived reference metrics and completed `VERIFICATION.md`.

### [2026-06-01 07:14] - core_claim_plus
- Sub-agent audit confirmed the scored path is a real scaled O-PTP
  implementation: inverse-CDF teacher sampling, auxiliary-conditioned causal
  one-call prediction, and teacher-prefix verification.
- Kept the arithmetic-coding `[ar+lin]` auxiliary embedding because Appendix E
  identifies it as the paper's selected structured embedding for experiments.
- Updated `scripts/run.sh` to regenerate baseline and method artifacts before
  scoring, then submitting a CPU test job to validate the fresh-run path.

### [2026-06-01 07:16] - core_claim_plus
- Completed final verification of problem/solution separation, scoring
  coefficients, reusable artifact-based evaluation, train/test split wiring,
  download portability, and milestone consistency.
- Confirmed `python validate.py --compare` passes; the remaining validate
  warnings are false positives because train/test evaluators read prediction
  artifacts and only regenerate reference targets from `data/`.
- Recorded a final `VERIFICATION.md` judgment: ready for gym use at
  `core_claim_plus`, limited to the scaled synthetic benchmark.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 16m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

### [2026-06-01 07:03] - core_claim_plus
- Completed a packaging audit of the scorecard and found the main
  `scoring/scores.json` omitted secondary metrics declared in
  `scoring/reference.json`.
- Updated the evaluator and score artifact so the official scorecard includes
  token accuracy, tail-token accuracy, prefix-8 success rate, and the
  parameter-count constraint.
- Submitting a CPU test job to rerun `scripts/evaluate.sh` inside the managed
  container because the login-node Python environment lacks `numpy`.

### [2026-06-01 07:08] - core_claim_plus
- Final verification confirmed the scaled O-PTP reproduction supports the core
  accepted-prefix claim and the restricted N=8 setting.
- Tightened method-agnostic documentation and made the main evaluator discover
  arbitrary compatible prediction artifacts under `checkpoints/`.
- Documented derived reference metrics and completed `VERIFICATION.md`.

### [2026-06-01 07:14] - core_claim_plus
- Sub-agent audit confirmed the scored path is a real scaled O-PTP
  implementation: inverse-CDF teacher sampling, auxiliary-conditioned causal
  one-call prediction, and teacher-prefix verification.
- Kept the arithmetic-coding `[ar+lin]` auxiliary embedding because Appendix E
  identifies it as the paper's selected structured embedding for experiments.
- Updated `scripts/run.sh` to regenerate baseline and method artifacts before
  scoring, then submitting a CPU test job to validate the fresh-run path.

### [2026-06-01 07:16] - core_claim_plus
- Completed final verification of problem/solution separation, scoring
  coefficients, reusable artifact-based evaluation, train/test split wiring,
  download portability, and milestone consistency.
- Confirmed `python validate.py --compare` passes; the remaining validate
  warnings are false positives because train/test evaluators read prediction
  artifacts and only regenerate reference targets from `data/`.
- Recorded a final `VERIFICATION.md` judgment: ready for gym use at
  `core_claim_plus`, limited to the scaled synthetic benchmark.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 20m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

### [2026-06-01 07:03] - core_claim_plus
- Completed a packaging audit of the scorecard and found the main
  `scoring/scores.json` omitted secondary metrics declared in
  `scoring/reference.json`.
- Updated the evaluator and score artifact so the official scorecard includes
  token accuracy, tail-token accuracy, prefix-8 success rate, and the
  parameter-count constraint.
- Submitting a CPU test job to rerun `scripts/evaluate.sh` inside the managed
  container because the login-node Python environment lacks `numpy`.

### [2026-06-01 07:08] - core_claim_plus
- Final verification confirmed the scaled O-PTP reproduction supports the core
  accepted-prefix claim and the restricted N=8 setting.
- Tightened method-agnostic documentation and made the main evaluator discover
  arbitrary compatible prediction artifacts under `checkpoints/`.
- Documented derived reference metrics and completed `VERIFICATION.md`.

### [2026-06-01 07:14] - core_claim_plus
- Sub-agent audit confirmed the scored path is a real scaled O-PTP
  implementation: inverse-CDF teacher sampling, auxiliary-conditioned causal
  one-call prediction, and teacher-prefix verification.
- Kept the arithmetic-coding `[ar+lin]` auxiliary embedding because Appendix E
  identifies it as the paper's selected structured embedding for experiments.
- Updated `scripts/run.sh` to regenerate baseline and method artifacts before
  scoring, then submitting a CPU test job to validate the fresh-run path.

### [2026-06-01 07:16] - core_claim_plus
- Completed final verification of problem/solution separation, scoring
  coefficients, reusable artifact-based evaluation, train/test split wiring,
  download portability, and milestone consistency.
- Confirmed `python validate.py --compare` passes; the remaining validate
  warnings are false positives because train/test evaluators read prediction
  artifacts and only regenerate reference targets from `data/`.
- Recorded a final `VERIFICATION.md` judgment: ready for gym use at
  `core_claim_plus`, limited to the scaled synthetic benchmark.

### [2026-11-11 08:00] - core_claim_plus (hardening)
- Hidden test now runs each submitted proposer itself (evaluator-owned one-call
  causal-transformer contract, weights-only loading, measured 1.10x parameter
  gate) on 16,384 control blocks held only by `eval/test/`, against a frozen
  teacher copy; failures score worst values in every metric; all artifacts on
  disk and the canonical slots are always scored.
- Reference retrained from final code (seed 13; bit-identical to the red-team
  seed-13 run): 12.136 vs 4.347 accepted tokens for the independent baseline,
  7.074 vs 3.057 N=8 correct tokens (hidden test).
- Longer training (+2.07 accepted, 15x seed range) and an in-cap 4-layer shape
  (+0.54) outrank the reference; all 16 tested shortcuts fail or tie exactly.
  See `HARDENING.md`.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 25m | **GPU**: 3.1h
- **Jobs**: 7 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 06:28] - method_runs
- Existing scaled O-PTP and independent baseline pipelines execute end-to-end.
- Previous CPU job produced real prediction artifacts and demonstrated the
  auxiliary-conditioning effect, but evaluation has since been hardened to
  regenerate held-out teacher targets from `data/`.

### [2026-06-01 06:30] - method_runs
- Added fixed held-out auxiliary evaluation and proposal-length scoring for
  N=8.
- Submitting a follow-up job to regenerate artifacts in the new format before
  claiming `core_claim` or `core_claim_plus` under the hardened evaluator.

### [2026-06-01 06:42] - core_claim_plus
- Job `5fc7a54b-b00` completed successfully under the hardened evaluator.
- O-PTP beat independent prediction at equal parameter count on the full
  synthetic auxiliary ablation: 12.189 vs 4.314 accepted tokens.
- The same effect held at restricted proposal length N=8, supporting the
  paper's claim across an additional compute setting.
- The scored N=16 diagnostic was removed because the synthetic independent
  baseline is not numerically comparable to the paper's taxi MTP row.

### [2026-06-01 07:03] - core_claim_plus
- Completed a packaging audit of the scorecard and found the main
  `scoring/scores.json` omitted secondary metrics declared in
  `scoring/reference.json`.
- Updated the evaluator and score artifact so the official scorecard includes
  token accuracy, tail-token accuracy, prefix-8 success rate, and the
  parameter-count constraint.
- Submitting a CPU test job to rerun `scripts/evaluate.sh` inside the managed
  container because the login-node Python environment lacks `numpy`.

### [2026-06-01 07:08] - core_claim_plus
- Final verification confirmed the scaled O-PTP reproduction supports the core
  accepted-prefix claim and the restricted N=8 setting.
- Tightened method-agnostic documentation and made the main evaluator discover
  arbitrary compatible prediction artifacts under `checkpoints/`.
- Documented derived reference metrics and completed `VERIFICATION.md`.

### [2026-06-01 07:14] - core_claim_plus
- Sub-agent audit confirmed the scored path is a real scaled O-PTP
  implementation: inverse-CDF teacher sampling, auxiliary-conditioned causal
  one-call prediction, and teacher-prefix verification.
- Kept the arithmetic-coding `[ar+lin]` auxiliary embedding because Appendix E
  identifies it as the paper's selected structured embedding for experiments.
- Updated `scripts/run.sh` to regenerate baseline and method artifacts before
  scoring, then submitting a CPU test job to validate the fresh-run path.

### [2026-06-01 07:16] - core_claim_plus
- Completed final verification of problem/solution separation, scoring
  coefficients, reusable artifact-based evaluation, train/test split wiring,
  download portability, and milestone consistency.
- Confirmed `python validate.py --compare` passes; the remaining validate
  warnings are false positives because train/test evaluators read prediction
  artifacts and only regenerate reference targets from `data/`.
- Recorded a final `VERIFICATION.md` judgment: ready for gym use at
  `core_claim_plus`, limited to the scaled synthetic benchmark.

### [2026-11-11 08:00] - core_claim_plus (hardening)
- Hidden test now runs each submitted proposer itself (evaluator-owned one-call
  causal-transformer contract, weights-only loading, measured 1.10x parameter
  gate) on 16,384 control blocks held only by `eval/test/`, against a frozen
  teacher copy; failures score worst values in every metric; all artifacts on
  disk and the canonical slots are always scored.
- Reference retrained from final code (seed 13; bit-identical to the red-team
  seed-13 run): 12.136 vs 4.347 accepted tokens for the independent baseline,
  7.074 vs 3.057 N=8 correct tokens (hidden test).
- Longer training (+2.07 accepted, 15x seed range) and an in-cap 4-layer shape
  (+0.54) outrank the reference; all 16 tested shortcuts fail or tie exactly.
  See `HARDENING.md`.

### [2026-11-16] - core_claim_plus (second hardening: training compute)
- Reference O-PTP and independent baseline retrained from the final code for
  the paper's 150k steps (batch 512, warmup + cosine, GPU): hidden test 20.035
  vs 4.347 accepted tokens, 7.830 vs 3.057 N=8 correct tokens.
- In-budget better methods now outrank more training: 4-layer +1.31 accepted
  (rank +0.51), [ar+NN] +0.93 (+0.37) vs 2x training +0.62 / +0.43
  (+0.24 / +0.19); seed range 0.30. The 76.8M-sequence-pass training budget is
  part of the contract. See `HARDENING.md`.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**
