# Evaluation

## What a method submits

One file per method: `checkpoints/<method>_proposer.pt`, written with
`data.proposer.save_proposer(path, model)` for a `data.proposer.ProposerNet`.
It holds a small config and a tensor state dict, nothing else. The evaluators
rebuild the network from their own frozen copy of that contract, load the
tensors with `torch.load(weights_only=True)`, count the parameters themselves,
and call the model once per block of 32 control values.

Fixed by the evaluator: vocabulary 16, horizon 32, causal self-attention,
learned positional embedding, pre-norm GELU encoder layers, one forward call.
Chosen in the config (`data.proposer.CONFIG_SPACE`):

| key | allowed |
|---|---|
| `hidden_size` | int 8-512, divisible by `heads` |
| `layers` | int 1-4 |
| `heads` | int 1-16 |
| `ff_mult` | int 1-8 (feed-forward width = hidden x ff_mult) |
| `aux_embedding` | how the control value enters position k: `ar` (its 32 binary digits), `th` (32 thresholds 1{u <= j/32}), `fl` (the scalar), `none` (ignored) |
| `aux_head` | `lin` (affine) or `nn` (2-layer GELU MLP) map of those features |
| `aux_offset` | `0`: position k sees controls 1..k; `1`: controls 1..k-1 |
| `decode` | `argmax` of the logits, or `pick` (inverse-CDF of the softmax at the position's control value) |

**Hard gates.** A run is failed, and receives the worst value of every scored
metric in both experiments (accepted 1, everything else 0), if its artifact is
missing, cannot be loaded as tensors only, has a config outside the table,
tensors that do not match the config or are not finite, non-finite logits, or
more than 377,960 parameters (1.10x the 343,600-parameter baseline). The slots
`o_ptp` and `independent_prediction` are always scored (failed if absent).
Sidecar JSON files, prediction arrays and score files written by a method are
never read.

## Metrics

Per block the evaluator compares the decoded proposal with the teacher's tokens
under the same control values.

- `avg_correct_tokens`: mean length of the longest matching prefix.
- `avg_accepted_tokens`: mean of `min(correct + 1, horizon)`; the verifier
  supplies one token at the first rejection (sequential correction).
- `token_accuracy`: fraction of all positions that match.
- `tail_token_accuracy`: match rate over the second half of the horizon.
- `prefix8_success_rate`: fraction of blocks whose first 8 tokens all match.

`synthetic_auxiliary_ablation` scores the full 32-token horizon;
`synthetic_restricted_compute_n8` scores the first 8 positions of the same
proposals (the model is causal, so this equals an 8-token call). The parameter
count and its ratio to the baseline are reported under `_constraints` in each
scorecard; they gate eligibility and are not ranked.

## Slices

- `bash scripts/evaluate_train.sh` (and `scripts/evaluate.sh`, which writes
  `scoring/scores.json`): the public development batch, 4,096 blocks, seed
  1013 (`data.ptp_synthetic.fixed_eval_batch`) -> `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: no arguments; a held-out batch of 16,384
  blocks known only to the evaluator -> `scoring/scores_test.json`.

Both slices draw control values from the same uniform distribution and use the
same code path, so an honest method scores alike on both; only the specific
control values differ.
