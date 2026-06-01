# Evaluation

The main metric is `avg_accepted_tokens`: the average number of tokens accepted
per verification step. For sequential correction this is
`min(correct_prefix + 1, sequence_length)`, because the verifier can provide one
replacement token at the first rejection.

The secondary benefit metric is `avg_correct_tokens`, the average longest prefix
of proposed tokens that exactly matches the reference samples under the same
random control values.

The constraint metric is `relative_param_multiplier`, the trainable parameter
count relative to the independent-prediction baseline. Lower is better; the
comparison keeps model scale fixed, so gains should come from better proposal
quality rather than more parameters.

The evaluator reads prediction artifacts from `checkpoints/*.npz`. Each artifact
must contain `auxiliaries` and `prediction` arrays with shape `[num_examples,
sequence_length]`; optional `confidence` and `teacher` arrays are ignored by the
score contract. `eval/` recomputes reference samples from `data/` and does not
import from `method/`.
