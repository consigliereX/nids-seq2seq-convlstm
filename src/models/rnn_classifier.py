from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        cell: str = "gru",
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        cell = cell.lower().strip()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(cell)
        if rnn_cls is None:
            raise ValueError(f"Unknown cell: {cell}")

        self.cell = cell
        self.bidirectional = bidirectional
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,d)
        out, h = self.rnn(x)
        # Use last timestep output (works for fixed T)
        last = out[:, -1, :]  # (B, hidden*dir)
        return self.head(last)
