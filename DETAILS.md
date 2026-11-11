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

The benchmark asks a method to propose a block of future discrete tokens for a
given prefix and a fixed stream of random control values. A separate verifier
samples the same future positions from its own conditional distributions using
the same control values. A proposed block is useful only up to the first token
that disagrees with the verifier.

A successful method should maximize the average length of the accepted prefix
under verifier comparison. It must not do so by simply increasing model size or
using a different evaluation target; the comparison includes a same-scale
multi-token baseline.

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

At evaluation, both systems propose all future tokens in one call. The teacher
verifier samples the reference sequence with the same auxiliaries, and the score
is the longest matching prefix plus the free verifier token, matching the
paper's sequential speculative decoding metric.

Deviation from the full paper scale: this workspace uses a small synthetic
autoregressive teacher rather than TinyLlama/CodeContests or Vicuna/SpecBench.
The algorithmic steps are preserved, but model size, data source, and training
length are scaled down to fit a compact reproducible job.

---

# Evaluation

The main metric is `avg_accepted_tokens`: the average number of tokens accepted
per verification step. For sequential correction this is
`min(correct_prefix + 1, sequence_length)`, because the verifier can provide one
replacement token at the first rejection.

The secondary benefit metric is `avg_correct_tokens`, the average longest prefix
of proposed tokens that exactly matches the reference samples under the same
random control values.

The constraint metric is `relative_param_multiplier`, the trainable parameter
count relative to the independent-prediction baseline. Lower is better; the
comparison keeps model scale fixed, so gains should come from better proposal
quality rather than more parameters.

The evaluator reads prediction artifacts from `checkpoints/*.npz`. Each artifact
must contain `auxiliaries` and `prediction` arrays with shape `[num_examples,
sequence_length]`; optional `confidence` and `teacher` arrays are ignored by the
score contract. `eval/` recomputes reference samples from `data/` and does not
import from `method/`.

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


---

# Reproduction Milestones

**Current: core_claim_plus**
