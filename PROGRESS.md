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
- A follow-up job is pending after the evaluation-hardening change. The previous
  numbers above came from the same O-PTP/independent algorithms but used an
  artifact format where stored teacher labels were trusted by eval.

## Remaining

- The next submitted job should regenerate artifacts with fixed held-out
  auxiliaries and write scores for the full synthetic auxiliary ablation plus
  restricted proposal lengths N=8 and N=16.
- Higher milestones beyond `core_claim_plus` would require a small
  language-model LoRA setup closer to the paper's CodeContests experiment or a
  C-PTP inverse-autoregressive training run.
- Continuation audit on 2026-06-01 verified `python validate.py --compare`
  passes and confirms the stored O-PTP result beats the independent baseline
  on both accepted and correct prefix length at equal parameter multiplier.
- Final continuation audit confirmed the artifacts contain real predictions
  from 1200-step method and baseline runs, each with 343,600 trainable
  parameters. The workspace is complete at `core_claim`.

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
- The bare login node does not expose Apptainer, so the hardened evaluator will
  be validated by the managed job rather than locally.

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
