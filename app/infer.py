from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure repo root is importable when running `python app/infer.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.hybrid_seq2seq_convlstm import HybridSeq2SeqConvLSTM
from src.utils.metrics import compute_metrics
from src.utils.seq_preproc import SeqPreprocStats, transform_with_stats


@dataclass(frozen=True)
class InferenceBundle:
    model: torch.nn.Module
    class_names: list[str]
    thresholds: dict[str, Any]
    stats: SeqPreprocStats
    device: torch.device


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path.as_posix()} is not a mapping")
    return data


def load_stats(path: Path) -> SeqPreprocStats:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return SeqPreprocStats(
        median=np.asarray(obj["median"], dtype=np.float32),
        mean=np.asarray(obj["mean"], dtype=np.float32),
        std=np.asarray(obj["std"], dtype=np.float32),
    )


def load_class_names(artifacts_dir: Path, input_npz: dict[str, Any] | None = None) -> list[str]:
    if input_npz is not None and "class_names" in input_npz:
        vals = input_npz["class_names"]
        return [str(x) for x in vals.tolist()]

    for name in ("pred_test.npz", "pred_val.npz"):
        p = artifacts_dir / name
        if p.exists():
            data = np.load(p, allow_pickle=True)
            if "class_names" in data:
                return [str(x) for x in data["class_names"].tolist()]

    raise FileNotFoundError(
        "Could not determine class_names. Provide them inside the input NPZ "
        "or keep pred_test.npz / pred_val.npz in the artifacts directory."
    )


def build_model_from_artifacts(
    artifacts_dir: Path,
    input_size: int,
    num_classes: int,
    class_names: list[str],
    device: torch.device,
) -> InferenceBundle:
    cfg = load_yaml(artifacts_dir / "config_used.yaml")
    p4 = cfg["phase4"]
    s2s_cfg = p4["seq2seq"]
    h_cfg = p4["hybrid"]

    ssh_idx = class_names.index("SSH-Patator") if "SSH-Patator" in class_names else -1

    model = HybridSeq2SeqConvLSTM(
        input_size=input_size,
        num_classes=num_classes,
        enc_hidden=int(s2s_cfg.get("hidden_size", 128)),
        enc_layers=int(s2s_cfg.get("num_layers", 1)),
        enc_bidir=bool(s2s_cfg.get("bidirectional", True)),
        enc_dropout=float(s2s_cfg.get("dropout", 0.2)),
        convlstm_hidden_channels=int(h_cfg.get("convlstm_hidden_channels", 32)),
        kernels=[int(k) for k in h_cfg.get("kernels", [3, 5, 7])],
        head_dropout=float(h_cfg.get("dropout", 0.2)),
        feature_map_mode=str(h_cfg.get("feature_map_mode", "none")),
        map_h=int(h_cfg.get("map_h", 1)),
        map_w=int(h_cfg.get("map_w", 1)),
        ssh_index=int(ssh_idx),
        ssh_aux=bool(h_cfg.get("ssh_aux", False)),
        ssh_gate_scale=float(h_cfg.get("ssh_gate_scale", 1.0)),
    ).to(device)

    ckpt = torch.load(artifacts_dir / "best.pt", map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    thr_path = artifacts_dir / "thresholds.json"
    thresholds = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.exists() else {}

    stats = load_stats(artifacts_dir / "preproc_stats.json")

    return InferenceBundle(
        model=model,
        class_names=class_names,
        thresholds=thresholds,
        stats=stats,
        device=device,
    )


def apply_ftp_min_threshold(
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    ftp_index: int,
    thr: float,
) -> np.ndarray:
    if thr <= 0:
        return y_pred
    y2 = y_pred.copy()
    mask = (y2 == ftp_index) & (y_proba[:, ftp_index] < thr)
    if not np.any(mask):
        return y2
    proba2 = y_proba[mask].copy()
    proba2[:, ftp_index] = -1.0
    y2[mask] = np.argmax(proba2, axis=1)
    return y2


def apply_ssh_margin_rule(
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    ssh_index: int,
    thr: float,
    margin: float,
) -> np.ndarray:
    if thr <= 0:
        return y_pred
    y2 = y_pred.copy()
    p_ssh = y_proba[:, ssh_index]
    p_max = y_proba.max(axis=1)
    mask = (p_ssh >= thr) & ((p_max - p_ssh) <= max(0.0, margin))
    y2[mask] = ssh_index
    return y2


def apply_decision_rules(
    y_pred_raw: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    thresholds: dict[str, Any],
) -> np.ndarray:
    y_pred = y_pred_raw.copy()

    if "FTP-Patator" in class_names:
        ftp_idx = class_names.index("FTP-Patator")
        ftp_thr = float(thresholds.get("ftp_min_threshold", 0.0))
        y_pred = apply_ftp_min_threshold(y_pred, y_proba, ftp_idx, ftp_thr)

    if "SSH-Patator" in class_names:
        ssh_idx = class_names.index("SSH-Patator")
        ssh_thr = float(thresholds.get("ssh_threshold", 0.0))
        ssh_margin = float(thresholds.get("ssh_margin", 0.0))
        y_pred = apply_ssh_margin_rule(y_pred, y_proba, ssh_idx, ssh_thr, ssh_margin)

    return y_pred


def run_inference(
    bundle: InferenceBundle,
    X: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = transform_with_stats(X, bundle.stats)
    ds = torch.from_numpy(X_scaled.astype(np.float32, copy=False))

    probs = []
    with torch.no_grad():
        for start in range(0, ds.shape[0], batch_size):
            xb = ds[start : start + batch_size].to(bundle.device)
            logits = bundle.model(xb)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.append(proba)

    y_proba = np.concatenate(probs, axis=0)
    y_pred_raw = np.argmax(y_proba, axis=1)
    y_pred = apply_decision_rules(y_pred_raw, y_proba, bundle.class_names, bundle.thresholds)
    return y_pred, y_proba


def save_predictions(
    out_dir: Path,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    y_true: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "row_id": np.arange(len(y_pred), dtype=np.int64),
        "pred_index": y_pred.astype(np.int64),
        "pred_label": [class_names[int(i)] for i in y_pred],
        "confidence": y_proba.max(axis=1).astype(np.float32),
    })

    if y_true is not None:
        df["true_index"] = y_true.astype(np.int64)
        df["true_label"] = [class_names[int(i)] for i in y_true]
        df["is_correct"] = (y_true == y_pred)

    for i, name in enumerate(class_names):
        safe = name.replace("/", "_").replace(" ", "_")
        df[f"proba_{safe}"] = y_proba[:, i].astype(np.float32)

    df.to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "n_samples": int(len(y_pred)),
        "class_names": class_names,
        "predicted_class_counts": df["pred_label"].value_counts().sort_index().to_dict(),
        "mean_confidence": float(df["confidence"].mean()),
    }

    if y_true is not None:
        metrics = compute_metrics(y_true, y_pred, y_proba, class_names=class_names)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary["accuracy"] = float(metrics["accuracy"])
        summary["macro_f1"] = float(metrics["macro_f1"])
        summary["weighted_f1"] = float(metrics["weighted_f1"])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run batch inference using the selected CICIDS2017 hybrid artifact bundle."
    )
    ap.add_argument("--input", required=True, help="Path to sequence NPZ with key X and optional y/class_names.")
    ap.add_argument(
        "--artifacts-dir",
        default="artifacts/CICIDS2017_best_hybrid",
        help="Directory containing best.pt, config_used.yaml, preproc_stats.json, thresholds.json.",
    )
    ap.add_argument("--output-dir", default="outputs/infer", help="Where predictions and summaries will be written.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    input_path = Path(args.input)
    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input NPZ: {input_path.as_posix()}")
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Missing artifacts dir: {artifacts_dir.as_posix()}")

    npz = np.load(input_path, allow_pickle=True)
    if "X" not in npz:
        raise KeyError("Input NPZ must contain key 'X' with shape (N, T, d).")

    X = np.asarray(npz["X"], dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N,T,d). Got shape={X.shape}")

    y_true = np.asarray(npz["y"], dtype=np.int64) if "y" in npz else None
    class_names = load_class_names(artifacts_dir, npz)

    bundle = build_model_from_artifacts(
        artifacts_dir=artifacts_dir,
        input_size=int(X.shape[-1]),
        num_classes=len(class_names),
        class_names=class_names,
        device=device,
    )

    y_pred, y_proba = run_inference(bundle=bundle, X=X, batch_size=int(args.batch_size))
    save_predictions(out_dir=out_dir, y_pred=y_pred, y_proba=y_proba, class_names=class_names, y_true=y_true)

    print(f"[OK] Wrote: {out_dir / 'predictions.csv'}")
    print(f"[OK] Wrote: {out_dir / 'summary.json'}")
    if y_true is not None:
        print(f"[OK] Wrote: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
