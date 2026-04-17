from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class SeqSplit:
    X: np.ndarray  # (N,T,d) float32
    y: np.ndarray  # (N,) int64
    feature_names: List[str]
    class_names: List[str]


def load_seq_npz(path: str | Path) -> SeqSplit:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ: {path.as_posix()}")

    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32, copy=False)
    y = data["y"].astype(np.int64, copy=False)

    feature_names = [str(x) for x in data["feature_names"].tolist()]
    class_names = [str(x) for x in data["class_names"].tolist()]

    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N,T,d). Got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D (N,). Got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y mismatch: X.N={X.shape[0]} y.N={y.shape[0]}")

    return SeqSplit(X=X, y=y, feature_names=feature_names, class_names=class_names)


def describe_split(split: SeqSplit) -> Dict[str, Any]:
    N, T, d = split.X.shape
    return {"N": int(N), "T": int(T), "d": int(d)}
