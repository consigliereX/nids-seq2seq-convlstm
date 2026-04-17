from __future__ import annotations
import torch
import torch.nn as nn

from src.models.seq2seq import Seq2SeqAutoencoder
from src.models.convlstm2d import ConvLSTM2D


class HybridSeq2SeqConvLSTM(nn.Module):
    """
    Hybrid model:
      - Seq2Seq encoder provides embeddings per timestep: (B,T,enc_dim)
      - ConvLSTM subnets operate on embedding maps shaped as (B,T,1,1,enc_dim)
      - Each subnet uses different kernel width along embedding axis (1 x k)
      - Pooled subnet outputs concatenated -> classifier
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        enc_hidden: int,
        enc_layers: int,
        enc_bidir: bool,
        enc_dropout: float,
        convlstm_hidden_channels: int,
        kernels: list[int],
        head_dropout: float,
        feature_map_mode: str = "none",
        map_h: int = 1,
        map_w: int = 1,
        ssh_index: int = -1,
        ssh_aux: bool = False,
        ssh_gate_scale: float = 1.0,
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
        enc_dim = enc_hidden * (2 if enc_bidir else 1)

        self.feature_map_mode = str(feature_map_mode).lower().strip()
        self.map_h = int(map_h)
        self.map_w = int(map_w)

        map_dim = self.map_h * self.map_w

        # If you request a real 2D map but enc_dim doesn't match, project enc_dim -> map_dim.
        if self.feature_map_mode in ("project_2d",):
            self.map_proj = nn.Linear(enc_dim, map_dim)
        else:
            self.map_proj = None

        # "reshape_2d" assumes enc_dim == map_dim (no projection).
        if self.feature_map_mode in ("reshape_2d",) and (enc_dim != map_dim):
            raise ValueError(f"reshape_2d requires enc_dim==map_h*map_w, got enc_dim={enc_dim}, map_dim={map_dim}")

        self.subnets = nn.ModuleList([
            ConvLSTM2D(c_in=1, c_h=convlstm_hidden_channels, kernel=(1, int(k)))
            for k in kernels
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = convlstm_hidden_channels * len(kernels)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(feat_dim, num_classes),
        )

        self.ssh_index = int(ssh_index)
        self.ssh_aux = bool(ssh_aux)
        self.ssh_gate_scale = float(ssh_gate_scale)

        # SSH auxiliary gate consumes the fused feature vector z (dim = feat_dim).
        self.ssh_gate = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(64, 1),
        ) if self.ssh_aux else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,d)
        emb = self.ae.encode(x)  # (B,T,enc_dim)

        # Feature-map upgrade: convert (B,T,enc_dim) into (B,T,C=1,H,W)
        # so ConvLSTM operates on meaningful spatial structure.
        z = emb  # [B,T,enc_dim]
        if self.feature_map_mode == "none":
            # legacy behavior: (B,T,1,1,enc_dim)
            emb_map = z.unsqueeze(2).unsqueeze(2)
        else:
            if self.map_proj is not None:
                z = self.map_proj(z)  # [B,T,map_h*map_w]
            B, T, _ = z.shape
            z = z.view(B, T, self.map_h, self.map_w)  # [B,T,H,W]
            emb_map = z.unsqueeze(2)                  # [B,T,1,H,W]

        feats = []
        for subnet in self.subnets:
            h_last = subnet(emb_map)          # (B, C_h, 1, enc_dim)
            pooled = self.pool(h_last).squeeze(-1).squeeze(-1)  # (B, C_h)
            feats.append(pooled)

        z = torch.cat(feats, dim=1)
        logits = self.head(z)

        if self.ssh_aux and (self.ssh_gate is not None) and (0 <= self.ssh_index < logits.shape[1]):
            gate_logit = self.ssh_gate(z).squeeze(1)  # (B,)
            logits[:, self.ssh_index] = logits[:, self.ssh_index] + (self.ssh_gate_scale * gate_logit)

        return logits
