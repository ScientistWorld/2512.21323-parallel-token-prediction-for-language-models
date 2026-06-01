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
- Hardened evaluation so `eval/` owns the deterministic held-out auxiliaries and
  teacher targets. Method artifacts now provide only predictions for the fixed
  auxiliary batch, plus confidence and metadata.

## Results

- Completed hardened CPU reproduction job `5fc7a54b-b00`; it regenerated
  method and baseline artifacts with fixed held-out auxiliaries and wrote
  `scoring/scores.json`.
- Full synthetic auxiliary ablation:
  - O-PTP: 12.189 average accepted tokens, 11.195 average correct tokens,
    parameter multiplier 1.0.
  - Independent prediction baseline: 4.314 average accepted tokens, 3.314
    average correct tokens, parameter multiplier 1.0.
- Restricted proposal length N=8:
  - O-PTP: 7.337 average accepted tokens, 7.087 average correct tokens.
  - Independent prediction: 3.883 average accepted tokens, 3.019 average
    correct tokens.
- Restricted proposal length N=16:
  - O-PTP: 11.401 average accepted tokens, 10.589 average correct tokens.
  - Independent prediction: 4.314 average accepted tokens, 3.314 average
    correct tokens.
- Paper references: Table 2 reports O-PTP 7.0 accepted tokens vs independent
  prediction 6.2 on CodeContests/TinyLlama; Table 10 reports O-PTP beating MTP
  at restricted proposal lengths. This reproduction uses a scaled synthetic
  autoregressive teacher but preserves the inverse-CDF auxiliary conditioning,
  one-call parallel prediction, and teacher-prefix verification computation.
- `python validate.py --compare` passes.

## Remaining

- Higher milestones beyond `core_claim_plus` would require a small
  language-model LoRA setup closer to the paper's CodeContests experiment, a
  C-PTP inverse-autoregressive training run, or partial quadratic decoding.
- The current environment is complete and reusable for the scaled core
  auxiliary-conditioning claim and restricted-compute ablations.

## Issues

- Full TinyLlama/Vicuna reproduction is too expensive for the first job because
  the paper trains LoRA adapters for 1M+ steps on large LLMs.
- Continuation audit found no `checkpoints/*.npz` artifacts and no
  `scoring/scores.json`; the previous submission did not leave visible job
  outputs in the workspace. The implementation and reference schema passed
  `python validate.py --reference-only`.
- Running `python -m eval.evaluate_ptp` in the bare login environment failed
  because `numpy` is not installed there. The container setup installs `numpy`
  and `torch`; script-level evaluation should be run inside the managed
  container or via submitted jobs.
- The bare login node does not expose Apptainer, so the hardened evaluator was
  validated by managed job `5fc7a54b-b00` rather than locally.

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
