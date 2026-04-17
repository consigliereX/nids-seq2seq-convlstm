from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def _majority_vote(labels: np.ndarray) -> str:
    # deterministic tie-breaker: highest count, then lexicographic
    vals, counts = np.unique(labels, return_counts=True)
    maxc = counts.max()
    winners = vals[counts == maxc]
    return str(sorted(winners)[0])


def _window_label(labels: np.ndarray, mode: str) -> str:
    if mode == "majority":
        return _majority_vote(labels)
    if mode == "any_attack":
        # BENIGN vs ATTACK
        return "ATTACK" if np.any(labels != "BENIGN") else "BENIGN"
    raise ValueError(f"Unknown label_mode: {mode}")


def _encode_labels(y_str: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    classes = sorted(list(np.unique(y_str)))
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[str(v)] for v in y_str], dtype=np.int64)
    return y, classes


def _time_split_by_order(order_first: np.ndarray, train_frac: float, val_frac: float) -> Dict[str, np.ndarray]:
    # Split by sequence start order (time-aware)
    n = len(order_first)
    # Sort indices by order_first (stable)
    idx_sorted = np.argsort(order_first, kind="mergesort")
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_idx = idx_sorted[:n_train]
    val_idx = idx_sorted[n_train:n_train + n_val]
    test_idx = idx_sorted[n_train + n_val:]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _class_counts(y: np.ndarray, class_names: List[str]) -> Dict[str, int]:
    counts = {c: 0 for c in class_names}
    vals, cts = np.unique(y, return_counts=True)
    for v, ct in zip(vals, cts):
        counts[class_names[int(v)]] = int(ct)
    return counts

def _stratified_3way_indices(idxs: np.ndarray, y: np.ndarray, train_frac: float, val_frac: float,
                            seed: int, min_per_split: int) -> Dict[str, np.ndarray]:
    """
    Deterministic stratified 3-way split over idxs using labels y[idxs].
    Guarantees (best effort) at least min_per_split samples per class per split when possible.
    """
    rng = np.random.default_rng(seed)
    y_sub = y[idxs]

    split = {"train": [], "val": [], "test": []}
    classes = np.unique(y_sub)

    for c in classes:
        c_idxs = idxs[y_sub == c]
        c_idxs = c_idxs.copy()
        rng.shuffle(c_idxs)

        n = len(c_idxs)
        if n == 1:
            split["train"].append(c_idxs[0])
            continue
        if n == 2:
            split["train"].append(c_idxs[0])
            split["val"].append(c_idxs[1])
            continue

        # base proportional counts
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val

        # enforce minimums if feasible
        if n >= 3 * min_per_split:
            n_train = max(n_train, min_per_split)
            n_val = max(n_val, min_per_split)
            n_test = max(n_test, min_per_split)

        # fix totals
        while n_train + n_val + n_test > n:
            # reduce the largest bucket
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        while n_train + n_val + n_test < n:
            # add to train by default
            n_train += 1

        a = 0
        b = n_train
        d = n_train + n_val

        split["train"].extend(c_idxs[a:b].tolist())
        split["val"].extend(c_idxs[b:d].tolist())
        split["test"].extend(c_idxs[d:].tolist())

    return {k: np.array(v, dtype=np.int64) for k, v in split.items()}

def main():
    cfg = load_yaml(Path("configs") / "base.yaml").raw
    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("reproducibility", {}).get("deterministic", True)))

    log_cfg = cfg.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "results/logs"))
    log_file = log_dir / "phase2_make_sequences.log"
    logger = setup_logger("PHASE2", level=str(log_cfg.get("level", "INFO")), log_file=str(log_file))

    scfg = cfg["sequences"]
    in_parquet = Path(scfg["input_parquet"])
    label_col = str(scfg.get("label_col", "Label"))
    order_col = str(scfg.get("order_col", "order_index"))

    T = int(scfg.get("seq_len", 20))
    S = int(scfg.get("stride", 5))
    label_mode = str(scfg.get("label_mode", "majority"))

    split_cfg = scfg.get("split", {})
    split_mode = str(split_cfg.get("mode", "global_time"))
    min_per_class_per_split = int(split_cfg.get("min_per_class_per_split", 1))
    source_file_col = str(split_cfg.get("source_file_col", "source_file"))
    train_frac = float(split_cfg.get("train_frac", 0.70))
    val_frac = float(split_cfg.get("val_frac", 0.15))
    test_frac = float(split_cfg.get("test_frac", 0.15))
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    out_dir = Path(scfg.get("output_dir", "data/processed/sequences"))
    out_stem = str(scfg.get("output_stem", f"cicids2017_seq_T{T}_S{S}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {in_parquet.as_posix()}")

    logger.info(f"Loading: {in_parquet.as_posix()}")
    df = pd.read_parquet(in_parquet)

    # Validate columns
    for col in [label_col, order_col]:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col}")

    # Sort by order_col to enforce deterministic ordering
    df = df.sort_values(order_col, kind="mergesort").reset_index(drop=True)

    # Feature columns: exclude label and order
    non_feature_cols = {label_col, order_col, "source_file"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    logger.info(f"Feature cols (d): {len(feature_cols)}")

    # Convert to numpy
    X_all = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_all = df[label_col].astype(str).to_numpy()
    order_all = df[order_col].to_numpy(dtype=np.int64, copy=True)

    n = len(df)
    if n < T:
        raise ValueError(f"Not enough rows ({n}) for sequence length T={T}")

    # Sliding windows
    X_seq = []
    y_seq = []
    order_first = []
    order_last = []

    # number of windows
    last_start = n - T
    for start in range(0, last_start + 1, S):
        end = start + T
        xw = X_all[start:end]
        lw = y_all[start:end]
        X_seq.append(xw)
        y_seq.append(_window_label(lw, label_mode))
        order_first.append(int(order_all[start]))
        order_last.append(int(order_all[end - 1]))

    X = np.stack(X_seq, axis=0)  # (N, T, d)
    y_str = np.array(y_seq, dtype=object)
    order_first = np.array(order_first, dtype=np.int64)
    order_last = np.array(order_last, dtype=np.int64)

    y, class_names = _encode_labels(y_str)

    logger.info(f"Built sequences X: {X.shape} (N,T,d) = ({X.shape[0]},{X.shape[1]},{X.shape[2]})")
    logger.info(f"Classes ({len(class_names)}): {class_names}")

    # Time-aware split by order_first
    split_idx = _time_split_by_order(order_first, train_frac=train_frac, val_frac=val_frac)

    if split_mode in ("per_source_file", "per_source_file_stratified"):
        if source_file_col not in df.columns:
            raise KeyError(f"split.mode={split_mode} requires column '{source_file_col}' in dataset.")

        # Assign each sequence to a source file by the sequence start row's source_file
        source_all = df[source_file_col].astype(str).to_numpy()
        start_indices = np.arange(0, (n - T) + 1, S, dtype=np.int64)
        seq_source = source_all[start_indices]

        split_idx = {"train": [], "val": [], "test": []}

        for src in np.unique(seq_source):
            idxs = np.where(seq_source == src)[0]  # indices into sequence arrays

            if split_mode == "per_source_file_stratified":
                # stratified-by-label within this file (reliable class coverage)
                s = _stratified_3way_indices(
                    idxs=idxs,
                    y=y,
                    train_frac=train_frac,
                    val_frac=val_frac,
                    seed=int(cfg.get("seed", 42)) + (hash(src) % 100000),
                    min_per_split=min_per_class_per_split,
                )
            else:
                # time-aware within this file
                idxs_sorted = idxs[np.argsort(order_first[idxs], kind="mergesort")]
                m = len(idxs_sorted)
                m_train = int(m * train_frac)
                m_val = int(m * val_frac)
                s = {
                    "train": idxs_sorted[:m_train],
                    "val": idxs_sorted[m_train:m_train + m_val],
                    "test": idxs_sorted[m_train + m_val:],
                }

            split_idx["train"].extend(s["train"].tolist())
            split_idx["val"].extend(s["val"].tolist())
            split_idx["test"].extend(s["test"].tolist())

        # Convert and deterministically sort by order_first for stable output
        split_idx = {k: np.array(v, dtype=np.int64) for k, v in split_idx.items()}
        for k in split_idx:
            split_idx[k] = split_idx[k][np.argsort(order_first[split_idx[k]], kind="mergesort")]

    else:
        # Default legacy behavior: global time split
        split_idx = _time_split_by_order(order_first, train_frac=train_frac, val_frac=val_frac)

    # Save splits to separate NPZs (explicit + convenient)
    outputs = {}
    for split_name, idxs in split_idx.items():
        out_npz = out_dir / f"{out_stem}_{split_name}.npz"
        np.savez_compressed(
            out_npz,
            X=X[idxs],
            y=y[idxs],
            order_first=order_first[idxs],
            order_last=order_last[idxs],
            feature_names=np.array(feature_cols, dtype=object),
            class_names=np.array(class_names, dtype=object),
            seq_source=seq_source[idxs],
        )
        outputs[split_name] = out_npz.as_posix()

        counts = _class_counts(y[idxs], class_names)
        logger.info(f"{split_name} -> N={len(idxs)} class_counts={counts}")

    # Save a manifest for reproducibility
    manifest = {
        "input": in_parquet.as_posix(),
        "T": T,
        "S": S,
        "label_mode": label_mode,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "splits": {
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
        },
        "outputs": outputs,
        "shapes": {
            "X_total": [int(X.shape[0]), int(X.shape[1]), int(X.shape[2])],
            "train_N": int(len(split_idx["train"])),
            "val_N": int(len(split_idx["val"])),
            "test_N": int(len(split_idx["test"])),
        },
    }
    (out_dir / f"{out_stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote manifest: {(out_dir / f'{out_stem}_manifest.json').as_posix()}")


if __name__ == "__main__":
    main()
