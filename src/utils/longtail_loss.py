from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LongTailStats:
    counts: np.ndarray  # shape [K]
    priors: np.ndarray  # shape [K]


def class_counts_and_priors(y: np.ndarray, K: int, eps: float = 1e-12) -> LongTailStats:
    counts = np.bincount(y.astype(np.int64, copy=False), minlength=K).astype(np.float64)
    priors = counts / max(float(counts.sum()), eps)
    priors = np.clip(priors, eps, 1.0)
    priors = priors / float(priors.sum())
    return LongTailStats(counts=counts.astype(np.int64), priors=priors.astype(np.float64))


class LogitAdjustedCrossEntropy(nn.Module):
    """
    Logit-Adjusted Cross-Entropy (train-only):
      logits_adj = logits - tau * log(prior)

    Intuition for long-tail: rare classes have small prior -> log(prior) very negative,
    subtracting it increases their effective logit during training (helps tail recall).
    """
    def __init__(self, priors: np.ndarray, tau: float = 1.0, label_smoothing: float = 0.0, eps: float = 1e-12):
        super().__init__()
        pri = np.array(priors, dtype=np.float64, copy=True)
        pri = np.clip(pri, eps, 1.0)
        pri = pri / float(pri.sum())
        log_prior = np.log(pri).astype(np.float32)

        self.register_buffer("log_prior", torch.tensor(log_prior, dtype=torch.float32))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B,K]
        adj = logits - (self.tau * self.log_prior.unsqueeze(0).to(logits.device))
        return F.cross_entropy(adj, target, label_smoothing=self.label_smoothing)


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax (train-only):
      p_k ∝ exp(logit_k) * n_k
    Equivalent to CE on (logits + log(counts)).

    Reference behavior: use raw logits at inference (no adjustment).
    """
    def __init__(self, counts: np.ndarray, label_smoothing: float = 0.0, eps: float = 1e-12):
        super().__init__()
        c = np.array(counts, dtype=np.float64, copy=True)
        c = np.clip(c, eps, None)
        log_c = np.log(c).astype(np.float32)
        self.register_buffer("log_counts", torch.tensor(log_c, dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adj = logits + self.log_counts.unsqueeze(0).to(logits.device)
        return F.cross_entropy(adj, target, label_smoothing=self.label_smoothing)
