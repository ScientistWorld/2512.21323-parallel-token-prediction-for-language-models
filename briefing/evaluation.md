# Evaluation

The main metric is `avg_accepted_tokens`: the average number of tokens accepted
per teacher-verification step. For sequential speculative correction this is
`min(correct_prefix + 1, sequence_length)`, because the verifier can provide one
replacement token at the first rejection.

The secondary benefit metric is `avg_correct_tokens`, the average longest prefix
of proposed tokens that exactly matches the teacher samples under the same
auxiliary variables.

The constraint metric is `relative_param_multiplier`, the trainable parameter
count relative to the independent-prediction baseline. Lower is better; the
paper's ablation keeps model scale fixed, so gains should come from better
conditioning rather than more parameters.

The evaluator reads prediction artifacts from `checkpoints/*.npz`. Each artifact
must contain `teacher` and `prediction` arrays with shape `[num_examples,
sequence_length]`; optional confidence arrays are ignored by the score contract.
`eval/` does not import from `method/`.
