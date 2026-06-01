# Scientist Constraints

## Model Scale
The scored proposer must stay within 1.10x of the independent baseline's
trainable parameter count for this workspace. The paper's ablation compares
models of the same scale, so simply increasing hidden size, layers, ensemble
count, or number of proposal models is not a valid improvement.

## Data Source
The scored training and evaluation data must come from `data/ptp_synthetic.py`
with the same teacher distribution and sequence length. External text corpora,
additional learned teachers, or regenerated labels from a different process are
not allowed for the scored experiment.

## Evaluation Protocol
The evaluator must score stored proposals against stored teacher sequences. It
must not regenerate the method's outputs inside `eval/`, call code from
`method/`, or use teacher labels during proposal selection beyond the verifier
comparison encoded in the metric.

## Compliance Checklist
- [ ] Trainable parameter count is no more than 1.10x the baseline count.
- [ ] Training and evaluation use the provided synthetic teacher only.
- [ ] `eval/` remains independent of `method/`.
- [ ] Scores are computed from artifacts written before evaluation.
