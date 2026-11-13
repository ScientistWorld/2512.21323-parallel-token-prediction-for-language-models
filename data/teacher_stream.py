"""Fast stream of fresh teacher samples for training on any torch device.

Control values u ~ U[0,1) come from a seeded numpy stream (the same draws
``data.ptp_synthetic.sample_batch`` would make), and the teacher's tokens are
picked by inverse-CDF lookup in the public teacher's CDF table on the target
device. ``TeacherStream.check`` verifies the lookup against the scalar teacher
definition, so the stream is the teacher, only faster on a GPU.

This is public training infrastructure. The evaluators never import it.
"""

from __future__ import annotations

import numpy as np
import torch

from data.ptp_synthetic import SEQ_LEN, VOCAB_SIZE, cdf_table, sample_teacher_from_u


class TeacherStream:
    """Yields (u, tokens) training batches of fresh teacher samples."""

    def __init__(self, batch_size: int, seed: int, device: torch.device | str,
                 chunk_batches: int = 16) -> None:
        self.batch_size = int(batch_size)
        self.chunk_batches = int(chunk_batches)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        table = np.array(cdf_table(SEQ_LEN, VOCAB_SIZE), dtype=np.float64)
        self.table = torch.from_numpy(table).to(self.device)
        self._buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

    def tokens(self, u: torch.Tensor) -> torch.Tensor:
        """Teacher tokens for control values ``u`` of shape [B, T] (any device)."""
        u64 = u.to(self.device).double()
        batch, seq_len = u64.shape
        out = torch.zeros(batch, seq_len, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch, dtype=torch.long, device=self.device)
        for k in range(seq_len):
            prev = out[:, k - 1] if k >= 1 else zeros
            prev2 = out[:, k - 2] if k >= 2 else zeros
            cdf = self.table[k, prev, prev2]
            out[:, k] = (cdf <= u64[:, k : k + 1]).sum(dim=1).clamp(max=VOCAB_SIZE - 1)
        return out

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._buffer:
            us = self.rng.random((self.chunk_batches * self.batch_size, SEQ_LEN), dtype=np.float32)
            u = torch.from_numpy(us).to(self.device)
            t = self.tokens(u)
            pairs = list(zip(u.split(self.batch_size), t.split(self.batch_size)))
            self._buffer = pairs[::-1]
        return self._buffer.pop()

    def check(self, num: int = 1024, seed: int = 20251213) -> None:
        """Raise if the device lookup disagrees with the scalar teacher."""
        us = np.random.default_rng(seed).random((num, SEQ_LEN), dtype=np.float32)
        expected = sample_teacher_from_u(us)
        got = self.tokens(torch.from_numpy(us)).cpu().numpy()
        if not np.array_equal(expected, got):
            raise RuntimeError("TeacherStream disagrees with data.ptp_synthetic teacher")
