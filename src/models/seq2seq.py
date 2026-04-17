from __future__ import annotations
import torch
import torch.nn as nn


class Seq2SeqAutoencoder(nn.Module):
    """
    Seq2Seq autoencoder for (B,T,d). Encoder produces sequence embeddings.
    Decoder reconstructs x for pretraining (MSE).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        cell: str = "gru",
    ) -> None:
        super().__init__()
        cell = cell.lower().strip()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(cell)
        if rnn_cls is None:
            raise ValueError(f"Unknown cell: {cell}")

        self.cell = cell
        self.bidirectional = bidirectional
        self.encoder = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        enc_dim = hidden_size * (2 if bidirectional else 1)

        self.decoder = rnn_cls(
            input_size=input_size,
            hidden_size=enc_dim,            # match encoder output dim
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.recon = nn.Linear(enc_dim, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,d) -> h: (B,T,enc_dim)
        h, _ = self.encoder(x)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # teacher-forcing style: feed original x to decoder, map decoder states to reconstruction
        h_enc = self.encode(x)            # (B,T,enc_dim)
        h_dec, _ = self.decoder(x)        # (B,T,enc_dim)
        x_hat = self.recon(h_dec)         # (B,T,d)
        return x_hat
