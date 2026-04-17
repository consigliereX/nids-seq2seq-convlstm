from __future__ import annotations

import argparse
import json
import time, os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.utils.config import load_yaml
from src.utils.env import get_torch_env, python_env_summary
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything
from src.utils.seq_io import load_seq_npz
from src.utils.seq_preproc import fit_train_only_stats, transform_with_stats, make_rf_features
from src.utils.metrics import compute_metrics, metrics_to_flat_rows, save_confusion_csv


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_config_copy(out_dir: Path, cfg_path: Path) -> None:
    out = out_dir / "config_used.yaml"
    out.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_env_snapshot(out_dir: Path) -> None:
    tenv = get_torch_env()
    snap = {
        "python": python_env_summary(),
        "torch": tenv.__dict__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(out_dir / "env_snapshot.json", snap)


def run_rf(cfg_raw: Dict[str, Any], run_id: str | None) -> Path:
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    p3 = cfg_raw["phase3"]
    rf_cfg = p3["rf"]
    out_root = Path(p3.get("output_dir", "results/phase3"))
    run_id = run_id or _now_run_id()
    run_dir = out_root / "rf" / f"{run_id}"
    _mkdir(run_dir)

    log_file = run_dir / "run.log"
    logger = setup_logger("PHASE3_RF", level=str(cfg_raw.get("logging", {}).get("level", "INFO")), log_file=str(log_file))

    _write_env_snapshot(run_dir)
    _write_config_copy(run_dir, Path("configs") / "base.yaml")

    train = load_seq_npz(p3["seq_npz"]["train"])
    val = load_seq_npz(p3["seq_npz"]["val"])
    test = load_seq_npz(p3["seq_npz"]["test"])

    class_names = train.class_names
    logger.info(f"Train X={train.X.shape} Val X={val.X.shape} Test X={test.X.shape}")
    logger.info(f"Classes={class_names}")

    # Preprocess: train-only stats on (N,T,d)
    stats = fit_train_only_stats(train.X)
    Xtr = transform_with_stats(train.X, stats)
    Xva = transform_with_stats(val.X, stats)
    Xte = transform_with_stats(test.X, stats)

    strategy = str(rf_cfg.get("strategy", "agg"))
    Ftr = make_rf_features(Xtr, strategy=strategy)
    Fva = make_rf_features(Xva, strategy=strategy)
    Fte = make_rf_features(Xte, strategy=strategy)

    logger.info(f"RF strategy={strategy} -> Train features shape={Ftr.shape}")

    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 200)),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        class_weight=rf_cfg.get("class_weight", None),
        random_state=int(cfg_raw.get("seed", 42)),
    )

    rf.fit(Ftr, train.y)

    def _eval(split_name: str, F: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        y_pred = rf.predict(F)
        y_proba = rf.predict_proba(F) if hasattr(rf, "predict_proba") else None

        metrics = compute_metrics(y_true=y, y_pred=y_pred, y_proba=y_proba, class_names=class_names)

        # Save preds
        np.savez_compressed(
            run_dir / f"pred_{split_name}.npz",
            y_true=y.astype(np.int64),
            y_pred=y_pred.astype(np.int64),
            y_proba=(y_proba.astype(np.float32) if y_proba is not None else np.zeros((len(y), len(class_names)), dtype=np.float32)),
            class_names=np.array(class_names, dtype=object),
        )

        cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
        save_confusion_csv(str(run_dir / f"confusion_{split_name}.csv"), cm, class_names, normalize=False)
        save_confusion_csv(str(run_dir / f"confusion_{split_name}_norm.csv"), cm, class_names, normalize=True)

        logger.info(f"{split_name}: macro_f1={metrics['macro_f1']:.6f} weighted_f1={metrics['weighted_f1']:.6f} acc={metrics['accuracy']:.6f}")
        return metrics

    m_val = _eval("val", Fva, val.y)
    m_test = _eval("test", Fte, test.y)

    # Save metrics.json + metrics.csv
    metrics_all = {"val": m_val, "test": m_test}
    _write_json(run_dir / "metrics.json", metrics_all)

    rows = []
    rows += metrics_to_flat_rows("val", m_val)
    rows += metrics_to_flat_rows("test", m_test)
    import pandas as pd
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)

    # Save model + preproc
    joblib.dump({"rf": rf, "strategy": strategy, "class_names": class_names}, run_dir / "model.joblib")
    joblib.dump({"stats": stats, "feature_names": train.feature_names}, run_dir / "preproc.joblib")

    logger.info(f"Wrote run_dir: {run_dir.as_posix()}")
    return run_dir


def run_rnn(cfg_raw: Dict[str, Any], run_id: str | None) -> Path:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from src.models.rnn_classifier import RNNClassifier
    from src.utils.seq_preproc import fit_train_only_stats, transform_with_stats

    p3 = cfg_raw["phase3"]
    rnn_cfg = p3["rnn"]
    out_root = Path(p3.get("output_dir", "results/phase3"))
    run_id = run_id or _now_run_id()
    run_dir = out_root / "rnn" / f"{run_id}"
    _mkdir(run_dir)

    log_file = run_dir / "run.log"
    logger = setup_logger("PHASE3_RNN", level=str(cfg_raw.get("logging", {}).get("level", "INFO")), log_file=str(log_file))

    _write_env_snapshot(run_dir)
    _write_config_copy(run_dir, Path("configs") / "base.yaml")

    train = load_seq_npz(p3["seq_npz"]["train"])
    val = load_seq_npz(p3["seq_npz"]["val"])
    test = load_seq_npz(p3["seq_npz"]["test"])
    class_names = train.class_names
    K = len(class_names)

    logger.info(f"Train X={train.X.shape} Val X={val.X.shape} Test X={test.X.shape}")
    logger.info(f"Classes={class_names}")

    stats = fit_train_only_stats(train.X)
    Xtr = transform_with_stats(train.X, stats)
    Xva = transform_with_stats(val.X, stats)
    Xte = transform_with_stats(test.X, stats)

    _write_json(run_dir / "preproc_stats.json", stats.to_jsonable())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(rnn_cfg.get("amp", True)) and (device.type == "cuda")

    class SeqDS(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X.astype(np.float32, copy=False))
            self.y = torch.from_numpy(y.astype(np.int64, copy=False))

        def __len__(self):
            return int(self.X.shape[0])

        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx]

    bs = int(rnn_cfg.get("batch_size", 512))
    nw = int(rnn_cfg.get("num_workers", 0))

    dl_tr = DataLoader(SeqDS(Xtr, train.y), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
    dl_va = DataLoader(SeqDS(Xva, val.y), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))
    dl_te = DataLoader(SeqDS(Xte, test.y), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))

    d = Xtr.shape[-1]
    model = RNNClassifier(
        input_size=d,
        hidden_size=int(rnn_cfg.get("hidden_size", 128)),
        num_layers=int(rnn_cfg.get("num_layers", 1)),
        num_classes=K,
        cell=str(rnn_cfg.get("cell", "gru")),
        bidirectional=bool(rnn_cfg.get("bidirectional", True)),
        dropout=float(rnn_cfg.get("dropout", 0.2)),
    ).to(device)

    # class weights (inverse frequency) with cap
    counts = np.bincount(train.y, minlength=K).astype(np.float64)
    inv = (counts.sum() / np.maximum(counts, 1.0))
    inv = inv / inv.mean()
    cap = float(rnn_cfg.get("class_weight_cap", 20.0))
    inv = np.clip(inv, 0.0, cap)
    w = torch.tensor(inv, dtype=torch.float32, device=device)

    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=float(rnn_cfg.get("lr", 1e-3)), weight_decay=float(rnn_cfg.get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = int(rnn_cfg.get("epochs", 15))
    patience = int(rnn_cfg.get("patience", 4))
    max_gn = float(rnn_cfg.get("max_grad_norm", 1.0))

    best_val = -1.0
    bad = 0
    hist_rows = []

    def _predict(dl: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        ys, yp, yp2 = [], [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                proba = torch.softmax(logits, dim=-1)
                pred = torch.argmax(proba, dim=-1)
                ys.append(yb.detach().cpu().numpy())
                yp.append(pred.detach().cpu().numpy())
                yp2.append(proba.detach().cpu().numpy())
        return np.concatenate(ys), np.concatenate(yp), np.concatenate(yp2)

    import pandas as pd

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                logits = model(xb)
                loss = crit(logits, yb)

            scaler.scale(loss).backward()
            if max_gn > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)

            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        y_true, y_pred, y_proba = _predict(dl_va)
        m = compute_metrics(y_true, y_pred, y_proba, class_names=class_names)

        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_macro_f1": m["macro_f1"],
            "val_weighted_f1": m["weighted_f1"],
            "val_acc": m["accuracy"],
        }
        hist_rows.append(row)
        pd.DataFrame(hist_rows).to_csv(run_dir / "train_history.csv", index=False)

        logger.info(f"epoch={ep} loss={row['train_loss']:.6f} val_macro_f1={row['val_macro_f1']:.6f} val_weighted_f1={row['val_weighted_f1']:.6f}")

        # early stopping on macro-F1
        if row["val_macro_f1"] > best_val:
            best_val = row["val_macro_f1"]
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val_macro_f1": best_val}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"Early stopping: patience={patience} reached.")
                break

        torch.save({"model": model.state_dict(), "epoch": ep, "best_val_macro_f1": best_val}, run_dir / "last.pt")

    # Load best + final eval (val + test)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    def _eval_save(split_name: str, dl: DataLoader) -> Dict[str, Any]:
        y_true, y_pred, y_proba = _predict(dl)
        metrics = compute_metrics(y_true, y_pred, y_proba, class_names=class_names)

        np.savez_compressed(
            run_dir / f"pred_{split_name}.npz",
            y_true=y_true.astype(np.int64),
            y_pred=y_pred.astype(np.int64),
            y_proba=y_proba.astype(np.float32),
            class_names=np.array(class_names, dtype=object),
        )

        cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
        save_confusion_csv(str(run_dir / f"confusion_{split_name}.csv"), cm, class_names, normalize=False)
        save_confusion_csv(str(run_dir / f"confusion_{split_name}_norm.csv"), cm, class_names, normalize=True)

        logger.info(f"{split_name}: macro_f1={metrics['macro_f1']:.6f} weighted_f1={metrics['weighted_f1']:.6f} acc={metrics['accuracy']:.6f}")
        return metrics

    m_val = _eval_save("val", dl_va)
    m_test = _eval_save("test", dl_te)

    _write_json(run_dir / "metrics.json", {"val": m_val, "test": m_test})

    rows = []
    rows += metrics_to_flat_rows("val", m_val)
    rows += metrics_to_flat_rows("test", m_test)
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)

    logger.info(f"Wrote run_dir: {run_dir.as_posix()}")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", choices=["rf", "rnn"], help="Which baseline to run")
    ap.add_argument("--run_id", default=None, help="Optional run id (default: timestamp)")
    args = ap.parse_args()

    if args.baseline == "rnn":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    cfg = load_yaml(Path("configs") / "base.yaml").raw
    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("reproducibility", {}).get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    if args.baseline == "rf":
        run_rf(cfg, run_id=args.run_id)
    else:
        run_rnn(cfg, run_id=args.run_id)


if __name__ == "__main__":
    main()
