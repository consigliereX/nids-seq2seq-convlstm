from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SeqPreprocStats:
    median: np.ndarray  # (d,)
    mean: np.ndarray    # (d,)
    std: np.ndarray     # (d,)

    def to_jsonable(self) -> Dict[str, list]:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def fit_train_only_stats(X_train: np.ndarray) -> SeqPreprocStats:
    """
    X_train: (N,T,d) with possible NaNs.
    Computes per-feature median for imputation, then mean/std for scaling after impute.
    """
    if X_train.ndim != 3:
        raise ValueError("X_train must be (N,T,d)")

    X2 = X_train.reshape(-1, X_train.shape[-1])  # (N*T, d)

    median = np.nanmedian(X2, axis=0).astype(np.float32)
    # fill NaN median (if a feature is all-NaN) with 0
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)

    X_imp = np.where(np.isfinite(X2), X2, median)  # (N*T, d)

    mean = X_imp.mean(axis=0).astype(np.float32)
    std = X_imp.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    return SeqPreprocStats(median=median, mean=mean, std=std)


def transform_with_stats(X: np.ndarray, stats: SeqPreprocStats) -> np.ndarray:
    """
    Apply median impute + standard scaling feature-wise, across all timesteps.
    """
    X2 = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)
    X_imp = np.where(np.isfinite(X2), X2, stats.median)
    X_scaled = (X_imp - stats.mean) / stats.std
    return X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2]).astype(np.float32, copy=False)


def make_rf_features(X_scaled: np.ndarray, strategy: str) -> np.ndarray:
    """
    X_scaled: (N,T,d)
    strategy:
      - "flatten": (N, T*d)
      - "agg":     (N, d*4) using mean/std/min/max across T
    """
    if strategy == "flatten":
        return X_scaled.reshape(X_scaled.shape[0], -1)

    if strategy == "agg":
        mean = X_scaled.mean(axis=1)
        std = X_scaled.std(axis=1)
        mn = X_scaled.min(axis=1)
        mx = X_scaled.max(axis=1)
        return np.concatenate([mean, std, mn, mx], axis=1)

    raise ValueError(f"Unknown RF feature strategy: {strategy}")
