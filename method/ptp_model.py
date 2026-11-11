"""O-PTP proposer configuration (paper Sec. 2.1, Eq. 5, Eq. 16, App. E.1).

The network family itself is the benchmark's artifact contract
(``data/proposer.py``): a causal transformer evaluated in one call per block.
One-Hot Parallel Token Prediction is the configuration below: every position k
receives the auxiliary u_k through the paper's [ar+lin] embedding (32 binary
arithmetic-coding digits of u, then a learned affine map), causal attention lets
position k see u_1..u_k, and the predicted token is the argmax of the one-hot
target distribution (Eq. 5).
"""

from __future__ import annotations

from data.proposer import ProposerNet, count_parameters, save_proposer  # noqa: F401

O_PTP_CONFIG = {
    "hidden_size": 96,
    "layers": 3,
    "heads": 4,
    "ff_mult": 4,
    "aux_embedding": "ar",
    "aux_head": "lin",
    "aux_offset": 0,
    "decode": "argmax",
}


def build_o_ptp(overrides: dict | None = None) -> ProposerNet:
    config = dict(O_PTP_CONFIG)
    config.update(overrides or {})
    return ProposerNet(config)
