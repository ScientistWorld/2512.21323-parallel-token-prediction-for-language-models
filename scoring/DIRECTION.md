# Research Direction

## Research Problem
This benchmark is about reducing the sequential bottleneck in autoregressive
discrete generation. The research question is how a proposer can generate
several dependent future tokens per model call while preserving compatibility
with teacher verification.

## Core Contribution
The paper's contribution is to condition parallel future-token prediction on
sampling auxiliaries so that later proposed tokens can depend on the sampled
outcomes of earlier proposed tokens. Improvements should remain in this family:
better ways to represent, train, or decode auxiliary-conditioned parallel token
proposers.

## Approach Scope
Work should improve multi-token proposal under teacher verification. Valid
changes include auxiliary embeddings, causal masking, training losses, curricula,
confidence calibration, and proposal selection, provided the method still
predicts multiple dependent future tokens from a common set of auxiliary random
variables.

## Out of Bounds
Replacing the task with ordinary autoregressive generation is out of bounds,
because it gives up the parallel proposal problem. Using a larger independent
predictor, an ensemble, a different teacher distribution, or a method that
selects proposals after looking at verifier labels is also out of bounds.
