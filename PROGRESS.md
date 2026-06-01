# Progress

## What Works

- Paper read and minimum core claim identified: auxiliary-conditioned O-PTP
  should produce longer teacher-verified accepted prefixes than independent
  multi-token prediction.
- Implemented a scaled O-PTP distillation pipeline on a synthetic
  autoregressive teacher with inverse-CDF auxiliaries and causal transformer
  conditioning.
- Implemented an independent-prediction baseline and an evaluator that reads
  stored prediction artifacts without importing from `method/`.

## Results

- No job results yet. The next submitted CPU job will train both small models
  and write `scoring/scores.json`.
- Paper Table 2 reference: O-PTP reports 7.0 accepted tokens; independent
  prediction reports 6.2 accepted tokens.

## Remaining

- Run the reproduction job.
- Inspect `scoring/scores.json`; if O-PTP does not beat the independent
  baseline by a meaningful margin, adjust training scale or synthetic task
  difficulty and resubmit.

## Issues

- Full TinyLlama/Vicuna reproduction is too expensive for the first job because
  the paper trains LoRA adapters for 1M+ steps on large LLMs.
- Continuation audit found no `checkpoints/*.npz` artifacts and no
  `scoring/scores.json`; the previous submission did not leave visible job
  outputs in the workspace. The implementation and reference schema passed
  `python validate.py --reference-only`.

## Deviations from Paper

- Paper setup: TinyLlama-1.1B on CodeContests and Vicuna-7B on ShareGPT/
  SpecBench with long LoRA training. Substitution: a small synthetic
  autoregressive teacher and small causal transformer. Reason: first-pass
  reproduction within the available budget while preserving the actual PTP
  computation.
- Paper proposal length: 16 tokens for CodeContests and larger settings
  elsewhere. Substitution: sequence length 32 for the synthetic benchmark.
- Algorithm not substituted: the code still uses inverse-CDF auxiliaries,
  O-PTP conditioning on `u_i ... u_k`, one-call parallel prediction, and
  teacher longest-prefix verification against the same auxiliaries.
