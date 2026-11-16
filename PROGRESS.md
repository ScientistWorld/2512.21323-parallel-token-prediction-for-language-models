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
- Updated the primary evaluator to report every metric declared in
  `scoring/reference.json`, including token accuracy, tail-token accuracy,
  prefix-8 success rate, and the parameter-count constraint.
- Sub-agent method audit confirmed the reproduction implements the scaled
  O-PTP computation rather than a surrogate: inverse-CDF teacher samples,
  auxiliary-conditioned causal one-call prediction, and verifier prefix scoring.
- `scripts/run.sh` now regenerates baseline and method artifacts before
  evaluation so submitted jobs do not depend on stale checkpoints.

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
- Official `scoring/scores.json` now includes the same metric set as the
  train/test split score files. The added full-split values are the mean of the
  equal-size train and test split reports from the completed job.
- Paper references: Table 2 reports O-PTP 7.0 accepted tokens vs independent
  prediction 6.2 on CodeContests/TinyLlama; Table 10 reports O-PTP beating MTP
  at restricted proposal length N=8. This reproduction uses a scaled synthetic
  autoregressive teacher but preserves the inverse-CDF auxiliary conditioning,
  one-call parallel prediction, and teacher-prefix verification computation.
- Validation note: an earlier scored N=16 synthetic diagnostic was removed from
  the official scorecard because its independent baseline is not numerically
  comparable to the paper's taxi MTP row. The corrected scored targets are the
  full synthetic auxiliary ablation and the N=8 restricted-compute setting.

## Remaining

- Higher milestones beyond `core_claim_plus` would require a small
  language-model LoRA setup closer to the paper's CodeContests experiment, a
  C-PTP inverse-autoregressive training run, or partial quadratic decoding.
- The current environment is complete and reusable for the scaled core
  auxiliary-conditioning claim and the N=8 restricted-compute ablation.
- Final verification found no critical gym issues; `scripts/run.sh` is set up to
  regenerate artifacts on future submissions.

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
- The login-node Python still lacks `numpy`, so the updated main evaluator is
  being rechecked through a managed CPU test job instead of direct local
  execution.
- A fidelity audit noted that the main text describes binary float embeddings,
  while Appendix E reports the arithmetic-coding `[ar+lin]` structured
  embedding selected for experiments. The local implementation uses the
  Appendix E arithmetic-coding embedding.

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

### [2026-11-11 08:00] - core_claim_plus (hardening)
- Hidden test now runs each submitted proposer itself (evaluator-owned one-call
  causal-transformer contract, weights-only loading, measured 1.10x parameter
  gate) on 16,384 control blocks held only by `eval/test/`, against a frozen
  teacher copy; failures score worst values in every metric; all artifacts on
  disk and the canonical slots are always scored.
- Reference retrained from final code (seed 13; bit-identical to the red-team
  seed-13 run): 12.136 vs 4.347 accepted tokens for the independent baseline,
  7.074 vs 3.057 N=8 correct tokens (hidden test).
- Longer training (+2.07 accepted, 15x seed range) and an in-cap 4-layer shape
  (+0.54) outrank the reference; all 16 tested shortcuts fail or tie exactly.
  See `HARDENING.md`.

### [2026-11-16] - core_claim_plus (second hardening: training compute)
- REVIEW.md: more training alone was the largest gain. The reference and the
  baseline now train for the paper's 150k steps (batch 512, AdamW 2e-3 peak,
  2k warmup, cosine), on a GPU through `data/teacher_stream.py`; the container
  installs CUDA torch 2.12.1 (cu126) in timeout-sized groups.
- Hidden test: O-PTP 20.035 vs independent 4.347 accepted tokens; N=8 correct
  7.830 vs 3.057. Learning curve 9.4k..300k steps: 15.97, 17.16, 18.36, 19.26,
  20.03, 20.65 (diminishing returns).
- Training compute is now a contract constraint (76.8M training-sequence
  passes, the reference's budget), checked from code and logs.
- `scoring/reference.json`: Table 2 numbers (16-token CodeContests scale) kept
  unchanged under `_paper_results`; `results` holds the gym's measured values.
