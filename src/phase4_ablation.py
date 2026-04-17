from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.seq_io import load_seq_npz
from src.utils.seq_preproc import fit_train_only_stats, transform_with_stats
from src.utils.metrics import compute_metrics, save_confusion_csv
from src.utils.longtail_loss import class_counts_and_priors


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _run_phase4_hybrid(out_dir: Path, seed: int, overrides: dict, run_id: str) -> int:
    _mkdir(out_dir)
    ovr_path = out_dir / "overrides.json"
    _write_json(ovr_path, overrides)

    cmd = [
        sys.executable, "-m", "src.phase4_hybrid",
        "--stage", "all",
        "--run_id", run_id,
        "--seed", str(seed),
        "--overrides_json", str(ovr_path),
        "--out_dir", str(out_dir),
    ]
    # Ensure deterministic flags consistent with your repo approach
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    print(f"[ABLATION] RUN seed={seed} out={out_dir.as_posix()}")
    print("[ABLATION] CMD:", " ".join(cmd))
    return subprocess.call(cmd, env=env)


def _rf_features(X: np.ndarray, strategy: str = "agg") -> np.ndarray:
    # X: (N,T,d)
    if strategy == "flatten":
        return X.reshape(X.shape[0], -1)
    # agg: mean/std/min/max across time (strong baseline for sequence tabular)
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    return np.concatenate([mu, sd, mn, mx], axis=1)


def _run_rf_baseline(out_dir: Path, seed: int, cfg_npz: dict, rf_cfg: dict) -> None:
    _mkdir(out_dir)

    train = load_seq_npz(cfg_npz["train"])
    val = load_seq_npz(cfg_npz["val"])
    test = load_seq_npz(cfg_npz["test"])
    class_names = train.class_names
    K = len(class_names)

    # Train-only preproc (match Phase4)
    stats = fit_train_only_stats(train.X)
    Xtr = transform_with_stats(train.X, stats)
    Xva = transform_with_stats(val.X, stats)
    Xte = transform_with_stats(test.X, stats)

    # Features
    strat = str(rf_cfg.get("strategy", "agg")).lower().strip()
    Ftr = _rf_features(Xtr, strategy=strat)
    Fva = _rf_features(Xva, strategy=strat)
    Fte = _rf_features(Xte, strategy=strat)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for RF baseline. Install: pip install scikit-learn"
        ) from e

    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 300)),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
        random_state=int(seed),
    )
    rf.fit(Ftr, train.y)

    def _eval(split: str, F: np.ndarray, y: np.ndarray) -> dict:
        proba = rf.predict_proba(F)
        pred = np.argmax(proba, axis=1)
        m = compute_metrics(y, pred, proba, class_names=class_names)

        cm = np.array(m["confusion_matrix"], dtype=np.int64)
        save_confusion_csv(str(out_dir / f"confusion_{split}.csv"), cm, class_names, normalize=False)
        save_confusion_csv(str(out_dir / f"confusion_{split}_norm.csv"), cm, class_names, normalize=True)

        # macro_f1_excl_ssh + explicit ssh stats if present
        if "SSH-Patator" in class_names:
            sidx = class_names.index("SSH-Patator")
            # compute PRF from cm
            tp = np.diag(cm).astype(np.float64)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            prec = tp / np.maximum(tp + fp, 1e-12)
            rec = tp / np.maximum(tp + fn, 1e-12)
            f1 = (2.0 * prec * rec) / np.maximum(prec + rec, 1e-12)
            keep = [i for i, n in enumerate(class_names) if n != "SSH-Patator"]
            m["macro_f1_excl_ssh"] = float(np.mean(f1[keep])) if keep else float("nan")
            m["ssh_precision"] = float(prec[sidx])
            m["ssh_recall"] = float(rec[sidx])
            m["ssh_f1"] = float(f1[sidx])
            m["ssh_support"] = int(cm.sum(axis=1)[sidx])

        np.savez_compressed(
            out_dir / f"pred_{split}.npz",
            y_true=y.astype(np.int64),
            y_pred=pred.astype(np.int64),
            y_proba=proba.astype(np.float32),
            class_names=np.array(class_names, dtype="U32"),
        )
        return m

    m_val = _eval("val", Fva, val.y)
    m_test = _eval("test", Fte, test.y)

    _write_json(out_dir / "metrics.json", {"val": m_val, "test": m_test})
    _write_json(out_dir / "run_meta.json", {
        "variant": "rf",
        "seed": int(seed),
        "rf_strategy": strat,
        "rf_cfg": rf_cfg,
        "train_counts": {class_names[i]: int(np.bincount(train.y, minlength=K)[i]) for i in range(K)},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def _collect_rows(run_dir: Path, variant: str, seed: int) -> list[dict]:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return []
    m = json.loads(mpath.read_text(encoding="utf-8"))
    rows = []
    for split in ("val", "test"):
        mm = m.get(split, {})
        rows.append({
            "variant": variant,
            "seed": int(seed),
            "split": split,
            "macro_f1": float(mm.get("macro_f1", np.nan)),
            "macro_f1_excl_ssh": float(mm.get("macro_f1_excl_ssh", np.nan)),
            "weighted_f1": float(mm.get("weighted_f1", np.nan)),
            "accuracy": float(mm.get("accuracy", np.nan)),
            "ssh_f1": float(mm.get("ssh_f1", np.nan)),
            "ssh_support": float(mm.get("ssh_support", np.nan)),
        })
    return rows


def _mean_std_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    d = df[df["split"] == split].copy()
    metrics = ["macro_f1", "macro_f1_excl_ssh", "ssh_f1", "weighted_f1", "accuracy"]
    g = d.groupby("variant")[metrics].agg(["mean", "std"]).reset_index()
    # Flatten columns
    cols = ["variant"]
    for m in metrics:
        cols += [f"{m}_mean", f"{m}_std"]
    out = pd.DataFrame(columns=cols)
    out["variant"] = g["variant"]
    for m in metrics:
        out[f"{m}_mean"] = g[(m, "mean")]
        out[f"{m}_std"] = g[(m, "std")]
    return out.sort_values(by="macro_f1_excl_ssh_mean", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_id", default=None)
    ap.add_argument("--cfg_train", default=None, help="Optional: path to a JSON with seq_npz/rf_cfg for RF baseline")
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (3–5 recommended)")
    ap.add_argument("--out_root", default="results/phase4/ablations")
    ap.add_argument("--run_rf", action="store_true", help="Include RF baseline (requires scikit-learn)")
    ap.add_argument("--variants", default="", help="Comma-separated variant names to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="Fast mode: reduce epochs/patience, disable threshold tuning")

    args = ap.parse_args()
    batch_id = args.batch_id or _now_id()
    out_root = Path(args.out_root) / batch_id
    _mkdir(out_root)

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    # These overrides assume your configs/base.yaml phase4 structure.
    # You can extend easily (tau sweeps, etc.) by adding more entries.
    variants: list[tuple[str, dict]] = [
        ("seq2seq_only", {"phase4": {"hybrid": {"variant": "seq2seq_only"}}}),
        ("convlstm_only_none", {"phase4": {"hybrid": {"variant": "convlstm_only", "feature_map_mode": "none", "map_h": 1, "map_w": 1}}}),
        ("hybrid_full_reshape", {"phase4": {"hybrid": {"variant": "hybrid", "feature_map_mode": "reshape_2d"}}}),
        ("hybrid_ablate_map_none", {"phase4": {"hybrid": {"variant": "hybrid", "feature_map_mode": "none", "map_h": 1, "map_w": 1}}}),
        ("hybrid_ablate_freeze_off", {"phase4": {"hybrid": {"variant": "hybrid", "freeze_encoder_epochs": 0}}}),
        ("hybrid_loss_balanced_softmax", {"phase4": {"hybrid": {"variant": "hybrid", "loss": "balanced_softmax"}}}),
        ("hybrid_loss_la_ce_tau_0p5", {"phase4": {"hybrid": {"variant": "hybrid", "loss": "la_ce", "la_tau": 0.5}}}),
        ("hybrid_loss_la_ce_tau_1p0", {"phase4": {"hybrid": {"variant": "hybrid", "loss": "la_ce", "la_tau": 1.0}}}),
        ("hybrid_loss_la_ce_tau_1p5", {"phase4": {"hybrid": {"variant": "hybrid", "loss": "la_ce", "la_tau": 1.5}}}),
    ]

    # Optional: run only selected variants
    if args.variants.strip():
        wanted = [x.strip() for x in args.variants.split(",") if x.strip()]
        wanted_set = set(wanted)
        variants = [(n, o) for (n, o) in variants if n in wanted_set]
        missing = wanted_set - {n for (n, _) in variants}
        if missing:
            print("[ABLATION] WARN: unknown variants ignored:", ",".join(sorted(missing)))

    # Optional: fast smoke mode
    quick_overrides = {}
    if args.quick:
        quick_overrides = {
            "phase4": {
                "seq2seq": {"pretrain_epochs": 1},
                "hybrid": {
                    "epochs": 6,
                    "patience": 2,
                    "auto_tune_thresholds": False,
                },
            }
        }

    # Optional RF baseline input (kept explicit to avoid depending on YAML parsing here)
    # Expected JSON shape:
    # {
    #   "seq_npz": {"train":"...npz","val":"...npz","test":"...npz"},
    #   "rf_cfg": {...}
    # }
    rf_cfg = None
    cfg_npz = None
    if args.run_rf:
        if not args.cfg_train:
            raise SystemExit("--run_rf requires --cfg_train JSON (seq_npz + rf_cfg).")
        j = json.loads(Path(args.cfg_train).read_text(encoding="utf-8"))
        cfg_npz = j["seq_npz"]
        rf_cfg = j.get("rf_cfg", {"strategy": "agg", "n_estimators": 300, "n_jobs": -1, "class_weight": "balanced_subsample"})

    all_rows: list[dict] = []

    # RF baseline
    if args.run_rf and cfg_npz and rf_cfg:
        for s in seeds:
            rdir = out_root / "rf" / f"seed_{s}"
            _run_rf_baseline(rdir, s, cfg_npz=cfg_npz, rf_cfg=rf_cfg)
            all_rows += _collect_rows(rdir, "rf", s)

    # DL variants
    for vname, overrides in variants:
        for s in seeds:
            rdir = out_root / vname / f"seed_{s}"
            ovr = _deep_merge(overrides, quick_overrides) if quick_overrides else overrides
            rc = _run_phase4_hybrid(rdir, s, overrides=ovr, run_id=f"{batch_id}_{vname}_seed{s}")
            if rc != 0:
                print(f"[ABLATION] WARN: run failed variant={vname} seed={s} rc={rc}")
            all_rows += _collect_rows(rdir, vname, s)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "runs_long.csv", index=False)

    # mean±std tables
    val_tbl = _mean_std_table(df, split="val")
    test_tbl = _mean_std_table(df, split="test")
    val_tbl.to_csv(out_root / "summary_val_mean_std.csv", index=False)
    test_tbl.to_csv(out_root / "summary_test_mean_std.csv", index=False)

    # quick markdown for paper drafts
    def _md_table(t: pd.DataFrame) -> str:
        # no external deps; simple markdown
        cols = list(t.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in t.iterrows():
            vals = []
            for c in cols:
                x = row[c]
                if isinstance(x, (float, np.floating)):
                    vals.append(f"{float(x):.4f}")
                else:
                    vals.append(str(x))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    (out_root / "summary_val_mean_std.md").write_text(_md_table(val_tbl), encoding="utf-8")
    (out_root / "summary_test_mean_std.md").write_text(_md_table(test_tbl), encoding="utf-8")

    print(f"[ABLATION] Wrote: {out_root.as_posix()}")


if __name__ == "__main__":
    main()
