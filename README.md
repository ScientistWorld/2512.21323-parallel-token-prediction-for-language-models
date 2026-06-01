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
autoregressive teacher.

The benchmark asks a method to propose a block of future discrete tokens for a
given prefix and a fixed stream of random auxiliary variables. A separate
teacher verifier samples the same future positions from its own conditional
distributions using the same auxiliary variables. A proposed block is useful
only up to the first token that disagrees with the teacher.

A successful method should maximize the average length of the accepted prefix
under teacher verification. It must not do so by simply increasing model size
or using a different evaluation target; the comparison is against an
independent multi-token predictor of the same scale.

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
per teacher-verification step. For sequential speculative correction this is
`min(correct_prefix + 1, sequence_length)`, because the verifier can provide one
replacement token at the first rejection.

The secondary benefit metric is `avg_correct_tokens`, the average longest prefix
of proposed tokens that exactly matches the teacher samples under the same
auxiliary variables.

The constraint metric is `relative_param_multiplier`, the trainable parameter
count relative to the independent-prediction baseline. Lower is better; the
paper's ablation keeps model scale fixed, so gains should come from better
conditioning rather than more parameters.

The evaluator reads prediction artifacts from `checkpoints/*.npz`. Each artifact
must contain `teacher` and `prediction` arrays with shape `[num_examples,
sequence_length]`; optional confidence arrays are ignored by the score contract.
`eval/` does not import from `method/`.

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


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
