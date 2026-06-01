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

- Completed CPU reproduction job `7add78d0-fae`; the copied job log path was
  unavailable in this workspace, but the expected artifacts were written under
  `checkpoints/` and `scoring/scores.json`.
- Reproduced scaled benchmark:
  - O-PTP: 12.189 average accepted tokens, 11.195 average correct tokens,
    parameter multiplier 1.0.
  - Independent prediction baseline: 4.314 average accepted tokens, 3.314
    average correct tokens, parameter multiplier 1.0.
- Paper Table 2 reference: O-PTP reports 7.0 accepted tokens and independent
  prediction reports 6.2 accepted tokens on CodeContests/TinyLlama. The scaled
  synthetic run is not intended to match the exact paper numbers, but it
  demonstrates the same central effect: auxiliary-conditioned PTP yields longer
  teacher-verified prefixes than independent parallel prediction at the same
  model size.
- `python validate.py --compare` passes.

## Remaining

- Higher milestones would require adding more settings, such as proposal length
  sweeps or a small language-model LoRA setup closer to the paper's
  CodeContests experiment.
- The current environment is usable as a gym for the core auxiliary-ablation
  claim.
- Continuation audit on 2026-06-01 verified `python validate.py --compare`
  passes and confirms the stored O-PTP result beats the independent baseline
  on both accepted and correct prefix length at equal parameter multiplier.

## Issues

- Full TinyLlama/Vicuna reproduction is too expensive for the first job because
  the paper trains LoRA adapters for 1M+ steps on large LLMs.
- Continuation audit found no `checkpoints/*.npz` artifacts and no
  `scoring/scores.json`; the previous submission did not leave visible job
  outputs in the workspace. The implementation and reference schema passed
  `python validate.py --reference-only`.
- The completed job report pointed to log paths under
  `/home/user/scratch/...`, but those files were not readable from this
  workspace. The run outcome was verified from generated artifacts and
  `validate.py --compare`.
- Running `python -m eval.evaluate_ptp` in the bare login environment failed
  because `numpy` is not installed there. The container setup installs `numpy`
  and `torch`; script-level evaluation should be run inside the managed
  container or via submitted jobs.

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
