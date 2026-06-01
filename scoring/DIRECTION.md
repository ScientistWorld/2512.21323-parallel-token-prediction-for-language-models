# Research Direction

## Research Problem
This benchmark is about reducing the sequential bottleneck in autoregressive
discrete generation. The research question is how a proposer can generate
several compatible future tokens per model call while preserving compatibility
with reference verification.

## Core Task
Methods should improve block proposal for dependent discrete sequences. A valid
proposal must be produced before verifier labels are known and must remain
compatible with the fixed reference distribution used by the evaluator.

## Approach Scope
Work should improve multi-token proposal under reference verification. Valid
changes include representation, objective design, curricula, confidence
calibration, and proposal selection, provided the method still predicts multiple
future tokens for the supplied prefix and random-control stream.

## Out of Bounds
Replacing the task with ordinary autoregressive generation is out of bounds,
because it gives up the parallel proposal problem. Using a larger independent
predictor, an ensemble, a different reference distribution, or a method that
selects proposals after looking at verifier labels is also out of bounds.
