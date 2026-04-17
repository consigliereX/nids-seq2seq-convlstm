from __future__ import annotations
import torch
import torch.nn as nn


class ConvLSTMCell2D(nn.Module):
    """
    ConvLSTM cell with 2D convolutions.
    Input per timestep: (B, C_in, H, W)
    Hidden state:       (B, C_h,  H, W)
    """
    def __init__(self, c_in: int, c_h: int, kernel: tuple[int, int]) -> None:
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(c_in + c_h, 4 * c_h, kernel_size=kernel, padding=padding)

        self.c_h = c_h

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch: int, H: int, W: int, device):
        h = torch.zeros(batch, self.c_h, H, W, device=device)
        c = torch.zeros(batch, self.c_h, H, W, device=device)
        return h, c


class ConvLSTM2D(nn.Module):
    def __init__(self, c_in: int, c_h: int, kernel: tuple[int, int]) -> None:
        super().__init__()
        self.cell = ConvLSTMCell2D(c_in=c_in, c_h=c_h, kernel=kernel)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B,T,C,H,W)
        returns last hidden: (B,C_h,H,W)
        """
        B, T, C, H, W = x_seq.shape
        h, c = self.cell.init_state(B, H, W, x_seq.device)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], (h, c))
        return h
