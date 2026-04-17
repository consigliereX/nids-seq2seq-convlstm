from __future__ import annotations

import torch
import torch.nn as nn

from src.models.convlstm2d import ConvLSTM2D


class ConvLSTMOnlyClassifier(nn.Module):
    """
    ConvLSTM-only classifier:
      x (B,T,d) -> map to (B,T,1,H,W) -> ConvLSTM subnets -> pooled concat -> head
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        convlstm_hidden_channels: int,
        kernels: list[int],
        head_dropout: float,
        feature_map_mode: str = "none",  # ["none","reshape_2d","project_2d"]
        map_h: int = 1,
        map_w: int = 1,
    ) -> None:
        super().__init__()
        self.feature_map_mode = str(feature_map_mode).lower().strip()
        self.map_h = int(map_h)
        self.map_w = int(map_w)
        map_dim = self.map_h * self.map_w

        if self.feature_map_mode in ("project_2d",):
            self.map_proj = nn.Linear(int(input_size), int(map_dim))
        else:
            self.map_proj = None

        if self.feature_map_mode in ("reshape_2d",) and (int(input_size) != int(map_dim)):
            raise ValueError(
                f"reshape_2d requires input_size==map_h*map_w, got input_size={input_size}, map_dim={map_dim}"
            )

        self.subnets = nn.ModuleList(
            [ConvLSTM2D(c_in=1, c_h=convlstm_hidden_channels, kernel=(1, int(k))) for k in kernels]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = convlstm_hidden_channels * len(kernels)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,d)
        z = x
        if self.feature_map_mode == "none":
            emb_map = z.unsqueeze(2).unsqueeze(2)  # (B,T,1,1,d)
        else:
            if self.map_proj is not None:
                z = self.map_proj(z)  # (B,T,map_h*map_w)
            B, T, _ = z.shape
            z = z.view(B, T, self.map_h, self.map_w)  # (B,T,H,W)
            emb_map = z.unsqueeze(2)                  # (B,T,1,H,W)

        feats = []
        for subnet in self.subnets:
            h_last = subnet(emb_map)
            pooled = self.pool(h_last).squeeze(-1).squeeze(-1)
            feats.append(pooled)

        zz = torch.cat(feats, dim=1)
        return self.head(zz)
