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

Both systems are instances of the benchmark's proposer contract
(`data/proposer.py`): O-PTP uses `aux_embedding="ar"`, `aux_head="lin"`,
`aux_offset=0`, `decode="argmax"`; the baseline uses `aux_embedding="none"`.
Both are hidden 96, 3 layers, 4 heads (343,600 parameters), trained with AdamW
(peak lr 2e-3 after a 2,000-step linear warmup, cosine decay to 0, weight
decay 0.01, clip 1.0) for the paper's 150,000 steps of 512 fresh teacher
samples each (seeds 13 and 17), which is the gym's whole training budget. A GPU
is used when available (`data/teacher_stream.py` samples the teacher on the
device); training the same recipe for 9.4k to 300k steps gives 16.0, 17.2,
18.4, 19.3, 20.0 and 20.7 accepted tokens on the hidden test, so the reference
sits where each doubling of training adds well under one token.

At evaluation the evaluator runs each proposer once per block on its own
control values; the teacher verifier samples the reference sequence with the
same auxiliaries, and the score is the longest matching prefix plus the free
verifier token, matching the paper's sequential speculative decoding metric.

Deviation from the full paper scale: this workspace uses a small synthetic
autoregressive teacher (order two, 16 tokens) rather than TinyLlama/CodeContests,
the taxi GPT-2 teacher or Vicuna/SpecBench, a 32-token horizon instead of 16 or
100, and a 343,600-parameter student trained from scratch for 150k steps of
batch 512 instead of 29M parameters for 150k steps of batch 32 (or LoRA
finetuning for 1.3M steps). The algorithmic steps are preserved; model size
and data source are scaled down, while the training length follows the paper's
step count.
