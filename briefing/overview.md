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
