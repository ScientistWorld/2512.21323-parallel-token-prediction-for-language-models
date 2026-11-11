"""Frozen copy of the proposer contract owned by the train-slice evaluator.

Identical to data/proposer.py when the gym was fixed. The train evaluator builds
every submitted proposer from this file, so edits to the public copy cannot
change the architecture, the parameter cap or the decoding that is scored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "ptp-proposer-v1"
VOCAB_SIZE = 16
SEQ_LEN = 32
AUX_BITS = 32
BASELINE_PARAM_COUNT = 343_600  # independent baseline: hidden 96, 3 layers, 4 heads
MAX_PARAM_COUNT = BASELINE_PARAM_COUNT * 11 // 10  # 1.10x cap = 377,960
MAX_FILE_BYTES = 32 * 1024 * 1024

INT_RANGES = {
    "hidden_size": (8, 512),
    "layers": (1, 4),
    "heads": (1, 16),
    "ff_mult": (1, 8),
}
CHOICES = {
    # How the control value u_k enters the network at position k.
    #   ar: 32 binary digits of u (arithmetic-coding style), th: 32 thresholds
    #   1{u <= j/32}, fl: the scalar u, none: a learned constant (u ignored).
    "aux_embedding": ("ar", "th", "fl", "none"),
    # Map from those features to the hidden size: linear or 2-layer GELU MLP.
    "aux_head": ("lin", "nn"),
    # 0: position k sees u_1..u_k; 1: position k sees u_1..u_{k-1}.
    "aux_offset": (0, 1),
    # argmax: token = argmax logits; pick: token = inverse-CDF of softmax at u_k.
    "decode": ("argmax", "pick"),
}
REQUIRED_KEYS = ("hidden_size", "layers", "heads", "aux_embedding")
DEFAULTS = {"ff_mult": 4, "aux_head": "lin", "aux_offset": 0, "decode": "argmax"}
CONFIG_SPACE = {"int_ranges": INT_RANGES, "choices": CHOICES, "defaults": DEFAULTS}


class ProposerContractError(ValueError):
    """The artifact does not satisfy the proposer contract."""


class ProposerConstraintError(ProposerContractError):
    """The artifact violates the scored parameter-count constraint."""


def validate_config(config: object) -> dict:
    if not isinstance(config, dict):
        raise ProposerContractError("config must be a dict")
    unknown = sorted(set(config) - set(INT_RANGES) - set(CHOICES))
    if unknown:
        raise ProposerContractError(f"unknown config keys: {unknown}")
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ProposerContractError(f"missing config keys: {missing}")
    cfg = dict(DEFAULTS)
    cfg.update(config)
    for key, (low, high) in INT_RANGES.items():
        value = cfg[key]
        if type(value) is not int or not low <= value <= high:
            raise ProposerContractError(f"{key} must be an int in [{low}, {high}]")
    for key, options in CHOICES.items():
        value = cfg[key]
        if type(value) not in (int, str) or value not in options:
            raise ProposerContractError(f"{key} must be one of {options}")
    if cfg["hidden_size"] % cfg["heads"]:
        raise ProposerContractError("hidden_size must be divisible by heads")
    return {key: cfg[key] for key in (*INT_RANGES, *CHOICES)}


class ProposerNet(nn.Module):
    """Causal transformer that maps a block of control values to token logits."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        cfg = validate_config(config)
        self.config = cfg
        hidden = cfg["hidden_size"]
        in_dim = 1 if cfg["aux_embedding"] == "fl" else AUX_BITS
        self.aux_proj = nn.Linear(in_dim, hidden)
        if cfg["aux_head"] == "nn":
            self.aux_proj2 = nn.Linear(hidden, hidden)
        self.mask_embedding = nn.Parameter(torch.zeros(hidden))
        self.pos_embedding = nn.Embedding(SEQ_LEN, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg["heads"],
            dim_feedforward=hidden * cfg["ff_mult"],
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg["layers"], enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, VOCAB_SIZE)

    def aux_features(self, auxiliaries: torch.Tensor) -> torch.Tensor:
        kind = self.config["aux_embedding"]
        if kind == "ar":
            powers = torch.pow(
                torch.tensor(2.0, device=auxiliaries.device),
                torch.arange(AUX_BITS, device=auxiliaries.device, dtype=auxiliaries.dtype),
            )
            bits = torch.remainder(auxiliaries.unsqueeze(-1) * powers, 1.0) <= 0.5
            return bits.to(auxiliaries.dtype)
        if kind == "th":
            thresholds = (
                torch.arange(1, AUX_BITS + 1, device=auxiliaries.device, dtype=auxiliaries.dtype)
                / AUX_BITS
            )
            return (auxiliaries.unsqueeze(-1) <= thresholds).to(auxiliaries.dtype)
        return auxiliaries.unsqueeze(-1)

    def forward(self, auxiliaries: torch.Tensor) -> torch.Tensor:
        if auxiliaries.dim() != 2 or auxiliaries.shape[1] > SEQ_LEN:
            raise ProposerContractError("auxiliaries must have shape [batch, <=32]")
        batch, seq_len = auxiliaries.shape
        positions = torch.arange(seq_len, device=auxiliaries.device)
        cfg = self.config
        if cfg["aux_embedding"] == "none":
            x = self.mask_embedding.view(1, 1, -1).expand(batch, seq_len, -1)
        else:
            x = self.aux_proj(self.aux_features(auxiliaries))
            if cfg["aux_head"] == "nn":
                x = self.aux_proj2(F.gelu(x))
            if cfg["aux_offset"] == 1:
                start = self.mask_embedding.view(1, 1, -1).expand(batch, 1, -1)
                x = torch.cat([start, x[:, :-1]], dim=1)
        x = x + self.pos_embedding(positions).unsqueeze(0)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=auxiliaries.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.transformer(x, mask=mask)
        return self.head(self.norm(x))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def decode(logits: torch.Tensor, auxiliaries: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "argmax":
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.double(), dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    tokens = (cdf <= auxiliaries.double().unsqueeze(-1)).sum(dim=-1)
    return tokens.clamp(max=logits.shape[-1] - 1)


def save_proposer(path: Path | str, model: ProposerNet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "config": dict(model.config),
        "state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)


def load_proposer(path: Path | str) -> tuple[ProposerNet, dict, int]:
    path = Path(path)
    if not path.is_file():
        raise ProposerContractError(f"{path.name}: not a regular file")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ProposerContractError(f"{path.name}: larger than {MAX_FILE_BYTES} bytes")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # malformed, pickled code, TorchScript, ...
        raise ProposerContractError(f"{path.name}: cannot be loaded as tensors only ({exc!r})") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ProposerContractError(f"{path.name}: payload is not a {FORMAT} dict")
    cfg = validate_config(payload.get("config"))
    model = ProposerNet(cfg)
    param_count = count_parameters(model)
    if param_count > MAX_PARAM_COUNT:
        raise ProposerConstraintError(
            f"{param_count} parameters exceeds cap {MAX_PARAM_COUNT}"
        )
    state = payload.get("state_dict")
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ProposerContractError("state_dict keys do not match the config")
    clean = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(expected[key].shape):
            raise ProposerContractError(f"tensor {key} has the wrong shape or type")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ProposerContractError(f"tensor {key} is not finite floating point")
        clean[key] = value.to(torch.float32)
    model.load_state_dict(clean, strict=True)
    model.eval()
    return model, cfg, param_count


def run_proposer(model: ProposerNet, auxiliaries: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """One forward call per block; returns int64 tokens of shape [N, T]."""
    outputs = []
    with torch.no_grad():
        for start in range(0, auxiliaries.shape[0], batch_size):
            u = torch.from_numpy(np.ascontiguousarray(auxiliaries[start : start + batch_size]))
            logits = model(u)
            if tuple(logits.shape) != (u.shape[0], u.shape[1], VOCAB_SIZE):
                raise ProposerContractError("logits have the wrong shape")
            if not bool(torch.isfinite(logits).all()):
                raise ProposerContractError("logits are not finite")
            outputs.append(decode(logits, u, model.config["decode"]).numpy().astype(np.int64))
    return np.concatenate(outputs, axis=0)
