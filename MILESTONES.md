# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-01 06:00] - none
- Read the paper markdown and extracted the core PTP distillation and teacher
  verification procedure.
- Implemented the scaled O-PTP and independent-prediction pipelines, but no
  compute job has completed yet, so no milestone above `none` is claimed.

### [2026-06-01 06:10] - none
- Audited the existing workspace. The implementation contains the paper's PTP
  computational steps at small scale, and eval remains separated from method.
- No job artifacts or scores were present, so the milestone remains `none`
  pending a real run.

### [2026-06-01 06:35] - core_claim
- Completed the CPU reproduction job for the scaled synthetic benchmark.
- O-PTP produced 12.189 average accepted tokens and 11.195 average correct
  tokens, compared with 4.314 and 3.314 for the independent predictor.
- Both methods use the same parameter multiplier, so the result supports the
  paper's core claim that meaningful auxiliary conditioning improves
  teacher-verified multi-token prediction without increasing model size.
- `python validate.py --compare` passes, including method/eval separation and
  score/reference schema checks.

## Stop Justification
- Completed at milestone `core_claim`.
