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
