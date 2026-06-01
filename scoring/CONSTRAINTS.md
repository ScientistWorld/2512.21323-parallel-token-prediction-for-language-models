# Scientist Constraints

## Model Scale
The scored proposer must stay within 1.10x of the independent baseline's
trainable parameter count for this workspace. Simply increasing hidden size,
layers, ensemble count, or number of proposal models is not a valid improvement.

## Data Source
The scored training and evaluation data must come from `data/ptp_synthetic.py`
with the same reference distribution and sequence length. External text corpora,
additional learned reference generators, or labels from a different process are
not allowed for the scored experiment.

## Evaluation Protocol
The evaluator regenerates the fixed held-out random-control batch and reference
targets from `data/ptp_synthetic.py`, then scores stored proposals against those
targets. Method artifacts must contain predictions for that exact control batch.
The evaluator must not regenerate the method's outputs inside `eval/`, call
code from `method/`, or use reference labels during proposal selection beyond
the verifier comparison encoded in the metric.

## Compliance Checklist
- [x] Trainable parameter count is no more than 1.10x the baseline count.
- [x] Training and evaluation use the provided synthetic reference generator only.
- [x] `eval/` remains independent of `method/`.
- [x] Scores are computed from artifacts written before evaluation.
