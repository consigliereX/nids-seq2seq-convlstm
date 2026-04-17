from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def _normalize_columns(cols: List[str]) -> List[str]:
    # CICIDS sometimes has leading/trailing spaces in headers
    return [c.strip() for c in cols]


def _read_one_csv(path: Path) -> pd.DataFrame:
    # low_memory=False avoids dtype fragmentation warnings on wide CSVs
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    df.columns = _normalize_columns(list(df.columns))
    return df


def _coerce_numeric_features(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = []
    for c in cols:
        # Try numeric coercion; non-numeric becomes NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
        numeric_cols.append(c)
    return df, numeric_cols


def _drop_constant_columns(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    kept = []
    dropped = []
    for c in feature_cols:
        series = df[c]
        # nunique(dropna=True) ignores NaNs; constant if <= 1 unique value
        if series.nunique(dropna=True) <= 1:
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def main():
    cfg = load_yaml(Path("configs") / "base.yaml").raw
    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("reproducibility", {}).get("deterministic", True)))

    log_cfg = cfg.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "results/logs"))
    log_file = log_dir / "phase1_preprocess.log"
    logger = setup_logger("PHASE1", level=str(log_cfg.get("level", "INFO")), log_file=str(log_file))

    dcfg = cfg["data"]["cicids2017"]
    raw_dir = Path(dcfg["raw_dir"])
    files = dcfg["files"]
    label_col = dcfg.get("label_col", "Label").strip()
    ts_col = dcfg.get("timestamp_col", "Timestamp").strip()
    leak_cols = [c.strip() for c in dcfg.get("leak_cols", [])]

    out_dir = Path(dcfg.get("output_dir", "data/processed"))
    out_stem = dcfg.get("output_stem", "cicids2017_clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + concat
    frames = []
    for fn in files:
        fp = raw_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing CICIDS2017 file: {fp.as_posix()}")
        logger.info(f"Reading: {fp.name}")
        df = _read_one_csv(fp)
        df["source_file"] = fp.name  # deterministic provenance for leakage-safe splitting
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Concat shape: {df.shape}")

    # Normalize column names again after concat
    df.columns = _normalize_columns(list(df.columns))

    # Basic required columns
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Columns: {list(df.columns)[:20]} ...")
    order_col = None
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
        order_col = ts_col
    elif "Flow Start Time" in df.columns:
        df["Flow Start Time"] = pd.to_datetime(df["Flow Start Time"], errors="coerce", infer_datetime_format=True)
        order_col = "Flow Start Time"
        logger.info("Using 'Flow Start Time' for ordering (not a feature).")
    else:
        # Deterministic fallback: preserve concatenation order as an explicit column
        df["order_index"] = np.arange(len(df), dtype=np.int64)
        order_col = "order_index"
        logger.warning("No timestamp found. Using 'order_index' for deterministic ordering (not a feature).")

    # Replace inf/-inf with NaN across all columns
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing labels
    before = len(df)
    df = df.dropna(subset=[label_col])
    logger.info(f"Dropped rows with missing label: {before - len(df)}")

    # Identify feature columns (exclude label + leak cols)
    exclude = set([label_col] + leak_cols)
    if "source_file" in df.columns:
        exclude.add("source_file")
    if order_col and order_col != label_col:
        exclude.add(order_col)
    exclude_present = [c for c in exclude if c in df.columns]
    logger.info(f"Excluding from features (present): {exclude_present}")

    # HARD DROP leakage columns from the saved dataset (prevents accidental feature usage later)
    leak_to_drop = [c for c in leak_cols if c in df.columns and c not in (label_col, order_col)]
    if leak_to_drop:
        df = df.drop(columns=leak_to_drop)
        logger.info(f"Dropped leakage columns from dataset: {leak_to_drop}")

    # Rebuild exclude_present after dropping
    exclude_present = [c for c in exclude if c in df.columns]

    # Coerce all non-excluded cols to numeric
    df, feature_cols = _coerce_numeric_features(df, exclude_cols=exclude_present)
    logger.info(f"Numeric feature cols: {len(feature_cols)}")

    # Drop constant columns (optional)
    dropped_constant = []
    if bool(dcfg.get("drop_constant_cols", True)):
        df, dropped_constant = _drop_constant_columns(df, feature_cols)
        feature_cols = [c for c in feature_cols if c not in dropped_constant]
        logger.info(f"Dropped constant cols: {len(dropped_constant)}")

    # Build class counts
    class_counts = df[label_col].value_counts(dropna=False).rename_axis("class").reset_index(name="count")

    # Save artifacts
    # Keep NaN in feature cols; imputation will be train-only in later phases
    out_parquet = out_dir / f"{out_stem}.parquet"
    out_csv = out_dir / f"{out_stem}.csv"

    # Write parquet (preferred)
    df.to_parquet(out_parquet, index=False)
    # Write CSV for easy inspection
    df.to_csv(out_csv, index=False)

    # Save metadata
    (out_dir / f"{out_stem}_features.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    class_counts.to_csv(out_dir / f"{out_stem}_class_counts.csv", index=False)

    report = {
        "input_files": files,
        "raw_dir": raw_dir.as_posix(),
        "rows": int(df.shape[0]),
        "cols_total": int(df.shape[1]),
        "label_col": label_col,
        "timestamp_col": ts_col,
        "leak_cols_present": exclude_present,
        "num_feature_cols": int(len(feature_cols)),
        "dropped_constant_cols": dropped_constant,
        "nan_counts_top20": (
            df[feature_cols].isna().sum().sort_values(ascending=False).head(20).to_dict()
            if feature_cols else {}
        ),
        "outputs": {
            "parquet": out_parquet.as_posix(),
            "csv": out_csv.as_posix(),
            "features": (out_dir / f"{out_stem}_features.txt").as_posix(),
            "class_counts": (out_dir / f"{out_stem}_class_counts.csv").as_posix(),
        },
    }
    (out_dir / f"{out_stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(f"Wrote: {out_parquet.as_posix()}")
    logger.info(f"Wrote: {out_csv.as_posix()}")
    logger.info(f"Wrote: {(out_dir / f'{out_stem}_features.txt').as_posix()}")
    logger.info(f"Wrote: {(out_dir / f'{out_stem}_class_counts.csv').as_posix()}")
    logger.info(f"Wrote: {(out_dir / f'{out_stem}_report.json').as_posix()}")


if __name__ == "__main__":
    main()
