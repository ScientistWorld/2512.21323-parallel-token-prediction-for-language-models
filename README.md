# Parallel Token Prediction for Language Models

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2512.21323"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2512.21323-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2512.21323](https://arxiv.org/abs/2512.21323) |
|:---|:---|
| **Domain** | Language model inference / speculative decoding |
| **TL;DR** | PTP feeds sampling randomness into a causal model so several |

## Summary

Autoregressive decoding is slow because each token normally requires a new
forward pass. Parallel Token Prediction moves the random variables used for
sampling into the model input, letting future tokens be predicted jointly while
remaining verifiable against an autoregressive teacher. The paper reports that
O-PTP improves accepted tokens and speedup over independent and speculative
decoding baselines.

## Key results of the paper

- Table 2: O-PTP obtains 7.0 accepted tokens versus 6.2 for independent
- Table 3: O-PTP reports 2.40x average speedup and 4.21 accepted tokens on

## Run it

```bash
pip install sciworld
sw research 2512.21323 --mode innovate --agent claude-code
sw evaluate 2512.21323
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

