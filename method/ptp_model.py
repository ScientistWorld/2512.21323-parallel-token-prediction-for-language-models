from __future__ import annotations

import torch
from torch import nn


class AuxiliaryTransformer(nn.Module):
    """Causal transformer over auxiliary variables for O-PTP-style prediction."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_size: int = 96,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        use_auxiliary: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.use_auxiliary = use_auxiliary
        self.aux_proj = nn.Linear(32, hidden_size)
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, auxiliaries: torch.Tensor) -> torch.Tensor:
        batch, seq_len = auxiliaries.shape
        positions = torch.arange(seq_len, device=auxiliaries.device)
        if self.use_auxiliary:
            powers = torch.pow(
                torch.tensor(2.0, device=auxiliaries.device),
                torch.arange(32, device=auxiliaries.device, dtype=auxiliaries.dtype),
            )
            # Arithmetic-coding style binary features from the paper's Eq. 16/App. E.1.
            aux_features = torch.remainder(auxiliaries.unsqueeze(-1) * powers, 1.0) <= 0.5
            x = self.aux_proj(aux_features.to(auxiliaries.dtype))
        else:
            x = self.mask_embedding.view(1, 1, -1).expand(batch, seq_len, -1)
        x = x + self.pos_embedding(positions).unsqueeze(0)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=auxiliaries.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.transformer(x, mask=mask)
        return self.head(self.norm(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
