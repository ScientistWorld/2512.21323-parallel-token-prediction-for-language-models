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
