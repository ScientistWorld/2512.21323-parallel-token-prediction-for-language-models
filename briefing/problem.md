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
