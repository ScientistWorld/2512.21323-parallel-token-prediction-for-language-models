# Experiment Overview

Both experiments are scored by the hidden test on 16,384 blocks of control
values that only the evaluator holds; the evaluator runs each submitted
proposer itself (one call per block) and compares its tokens with the
reference generator's under the same control values. Model size is a hard
eligibility gate (at most 1.10x the same-scale baseline), so neither experiment
can be improved by adding parameters.

## synthetic_auxiliary_ablation
Establishes whether a single-call block proposer for dependent future tokens
produces longer verifier-accepted prefixes over a 32-token horizon than a
same-scale system that predicts future tokens independently. Besides the
accepted and correct prefix lengths, it checks that quality holds across all
positions and in the second half of the horizon, and how often the first eight
tokens are all right.

## synthetic_restricted_compute_n8
The same comparison when only eight tokens are proposed per call. It checks
whether the advantage persists under a short proposal budget, where the
achievable maximum is eight tokens per verification step.
