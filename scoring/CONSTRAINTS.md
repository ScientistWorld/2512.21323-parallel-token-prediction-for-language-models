# Scientist Constraints

## Model Scale
The scored proposer must have at most 377,960 parameters (1.10x the 343,600
of the independent baseline). The evaluator counts them on the network it
builds; claims in sidecar files are ignored. Larger models score as failed.

## Single-Call Proposer Contract
A method submits `checkpoints/<name>_proposer.pt` written with
`data.proposer.save_proposer`: a config from `data.proposer.CONFIG_SPACE` and
the matching tensors. The evaluator builds a causal transformer of at most four
layers from it and calls it once per block. No other code, module, TorchScript
or pickled object is executed; anything else scores as failed.

## Data Source
Training may use any data, but the target is the reference generator in
`data/ptp_synthetic.py` with its 16-token vocabulary and 32-token horizon. The
evaluators hold frozen copies of that generator; editing the public copy does
not change what is scored.

## Evaluation Protocol
The hidden test generates its own control values and targets, runs every
proposer artifact on disk, and recomputes every number. It takes no arguments
and does not read anything a method writes besides the proposer artifacts.

## Compliance Checklist
- [x] Measured parameter count is no more than 1.10x the baseline count.
- [x] Proposer is a `ProposerNet` config + tensors (one forward call per block).
- [x] `eval/` remains independent of `method/`; `eval/train` and `eval/test`
      share no code.
- [x] Scores are computed by the evaluator from the submitted proposer.
