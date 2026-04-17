from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def confusion_to_fpr(cm: np.ndarray) -> np.ndarray:
    """
    One-vs-rest FPR per class:
      FPR_i = FP_i / (FP_i + TN_i)
    """
    K = cm.shape[0]
    total = cm.sum()
    fpr = np.zeros(K, dtype=np.float64)

    for i in range(K):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - (tp + fp + fn)
        denom = fp + tn
        fpr[i] = (fp / denom) if denom > 0 else 0.0
    return fpr


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: List[str],
) -> Dict[str, Any]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fpr = confusion_to_fpr(cm)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = []
    for idx, name in enumerate(class_names):
        r = report.get(name, {})
        per_class.append(
            {
                "class": name,
                "precision": float(r.get("precision", 0.0)),
                "recall": float(r.get("recall", 0.0)),
                "f1": float(r.get("f1-score", 0.0)),
                "support": int(r.get("support", 0)),
                "fpr": float(fpr[idx]),
            }
        )

    out = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "has_proba": bool(y_proba is not None),
    }
    return out


def metrics_to_flat_rows(split: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {
            "split": split,
            "metric": "macro_f1",
            "value": metrics["macro_f1"],
            "class": "",
        },
        {
            "split": split,
            "metric": "weighted_f1",
            "value": metrics["weighted_f1"],
            "class": "",
        },
        {
            "split": split,
            "metric": "accuracy",
            "value": metrics["accuracy"],
            "class": "",
        },
    ]
    per = metrics.get("per_class", [])

    # Compatibility:
    # - old format: list[{"class": name, "f1":..., "fpr":...}]
    # - new format: dict[name] -> {"f1":..., "fpr":...}
    if isinstance(per, dict):
        per_iter = [{"class": k, **(v or {})} for k, v in per.items()]
    else:
        per_iter = per

    for c in per_iter:
        rows.append({"split": split, "metric": "f1", "value": c.get("f1", 0.0), "class": c.get("class", "")})
        rows.append({"split": split, "metric": "fpr", "value": c.get("fpr", 0.0), "class": c.get("class", "")})
    return rows


def save_confusion_csv(out_path: str, cm: np.ndarray, class_names: List[str], normalize: bool = False) -> None:
    cm2 = cm.astype(np.float64)
    if normalize:
        row_sums = cm2.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        cm2 = cm2 / row_sums

    df = pd.DataFrame(cm2, index=class_names, columns=class_names)
    df.to_csv(out_path, index=True)
