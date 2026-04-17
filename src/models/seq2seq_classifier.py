from __future__ import annotations

import torch
import torch.nn as nn

from src.models.seq2seq import Seq2SeqAutoencoder


class Seq2SeqClassifier(nn.Module):
    """
    Seq2Seq-only classifier:
      x -> Seq2Seq encoder embeddings (B,T,enc_dim) -> pooling -> linear head
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        enc_hidden: int,
        enc_layers: int,
        enc_bidir: bool,
        enc_dropout: float,
        head_dropout: float,
        pooling: str = "mean",  # ["mean", "last"]
    ) -> None:
        super().__init__()
        self.ae = Seq2SeqAutoencoder(
            input_size=input_size,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            bidirectional=enc_bidir,
            dropout=enc_dropout,
            cell="gru",
        )
        self.pooling = str(pooling).lower().strip()
        enc_dim = enc_hidden * (2 if enc_bidir else 1)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(enc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.ae.encode(x)  # (B,T,enc_dim)
        if self.pooling == "last":
            z = emb[:, -1, :]
        else:
            z = emb.mean(dim=1)
        return self.head(z)
