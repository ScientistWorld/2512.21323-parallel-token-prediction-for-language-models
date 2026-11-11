# Research Direction

## Research Problem
This benchmark is about reducing the sequential bottleneck in autoregressive
discrete generation: how can a model generate several compatible future tokens
in one call while staying verifiable against a reference generator?

## Core Task
Train a single-call block proposer, within a fixed model family and parameter
budget, whose proposals match the reference generator's samples for as many
leading positions as possible, for control values it has never seen.

## Approach Scope
Everything about how the proposer is obtained is open: training objective,
training data and its source, curricula over positions or horizons, sampling
of training control values, optimisation, schedules and training length,
initialisation, distillation or self-training schemes, and the choice of shape
and input/decoding options inside the proposer contract.

## Out of Bounds
Running the reference generator token by token at inference (ordinary
autoregressive generation), exceeding the parameter cap, executing code other
than the evaluator's network, or changing the reference generator or the
evaluation protocol. The contract enforces these; attempts score as failed.
