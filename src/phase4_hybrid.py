from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler

from src.utils.config import load_yaml
from src.utils.env import get_torch_env, python_env_summary
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything
from src.utils.seq_io import load_seq_npz
from src.utils.seq_preproc import fit_train_only_stats, transform_with_stats
from src.utils.metrics import compute_metrics, metrics_to_flat_rows, save_confusion_csv

from src.models.seq2seq import Seq2SeqAutoencoder
from src.models.hybrid_seq2seq_convlstm import HybridSeq2SeqConvLSTM
from src.models.seq2seq_classifier import Seq2SeqClassifier
from src.models.convlstm_only import ConvLSTMOnlyClassifier

from src.utils.longtail_loss import (
    class_counts_and_priors,
    LogitAdjustedCrossEntropy,
    BalancedSoftmaxLoss,
)

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (optionally with class weights and label smoothing).
    - gamma: focusing parameter
    - weight: same semantics as CrossEntropyLoss(weight=...)
    - label_smoothing: small smoothing can help generalization; default 0 for focal runs
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B,K], target: [B]
        logp = torch.log_softmax(logits, dim=-1)  # [B,K]
        # NLL for the true class
        nll = torch.nn.functional.nll_loss(logp, target, weight=self.weight, reduction="none")  # [B]

        if self.label_smoothing > 0.0:
            # -mean log prob across classes
            smooth = -logp.mean(dim=-1)  # [B]
            nll = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth

        pt = torch.exp(logp.gather(1, target.view(-1, 1)).squeeze(1)).clamp(1e-8, 1.0)  # [B]
        focal = (1.0 - pt).pow(self.gamma) * nll
        return focal.mean()


def _macro_f1_excluding_from_cm(cm: np.ndarray, class_names: list[str], exclude: str) -> float:
    """
    Macro-F1 computed from confusion matrix, excluding `exclude` class.
    This avoids sklearn dependency and matches artifact-driven evaluation.
    """
    cm = cm.astype(np.float64, copy=False)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / np.maximum(tp + fn, 1e-12)
    f1 = (2.0 * prec * rec) / np.maximum(prec + rec, 1e-12)

    keep = [i for i, n in enumerate(class_names) if n != exclude]
    if not keep:
        return float("nan")
    return float(np.mean(f1[keep]))

def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_env_snapshot(out_dir: Path) -> None:
    tenv = get_torch_env()
    snap = {
        "python": python_env_summary(),
        "torch": tenv.__dict__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(out_dir / "env_snapshot.json", snap)

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_json_if_exists(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"override json not found: {p.as_posix()}")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _per_class_prf_from_cm(cm: np.ndarray, class_names: list[str]) -> dict:
    cm = cm.astype(np.float64, copy=False)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / np.maximum(tp + fn, 1e-12)
    f1 = (2.0 * prec * rec) / np.maximum(prec + rec, 1e-12)

    out = {}
    for i, name in enumerate(class_names):
        out[str(name)] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return out

class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = None if y is None else torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def _class_weights(y: np.ndarray, K: int, cap: float, device) -> torch.Tensor:
    counts = np.bincount(y, minlength=K).astype(np.float64)
    inv = (counts.sum() / np.maximum(counts, 1.0))
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.0, cap)
    return torch.tensor(inv, dtype=torch.float32, device=device)

def _make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(y)
    counts = np.maximum(counts, 1)
    class_w = 1.0 / counts
    sample_w = class_w[y].astype(np.float64)
    sample_w = sample_w / sample_w.mean()
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
    )

def _predict(model: nn.Module, dl: DataLoader, device, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _apply_ftp_min_threshold(y_pred: np.ndarray, y_proba: np.ndarray, ftp_index: int, thr: float) -> np.ndarray:
    """
    If argmax predicts FTP but P(FTP) < thr, demote FTP to the best non-FTP class.
    This targets the 'BENIGN -> FTP' false-positive explosion.
    """
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


def _apply_ssh_margin_rule(
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    ssh_index: int,
    thr: float,
    margin: float,
) -> np.ndarray:
    """
    Promote SSH only if:
      - P(SSH) >= thr
      - and SSH is within `margin` of the best class probability
    This avoids 'force SSH everywhere' false positives.
    """
    if thr <= 0:
        return y_pred
    y2 = y_pred.copy()
    p_ssh = y_proba[:, ssh_index]
    p_max = y_proba.max(axis=1)
    mask = (p_ssh >= thr) & ((p_max - p_ssh) <= max(0.0, margin))
    y2[mask] = ssh_index
    return y2


def _apply_decision_rules(y_pred: np.ndarray, y_proba: np.ndarray, class_names: list[str], h_cfg: dict) -> np.ndarray:
    y2 = y_pred

    # FTP demotion
    if "FTP-Patator" in class_names:
        ftp_idx = class_names.index("FTP-Patator")
        ftp_thr = float(h_cfg.get("ftp_min_threshold", 0.0))
        y2 = _apply_ftp_min_threshold(y2, y_proba, ftp_idx, ftp_thr)

    # SSH promotion (margin rule)
    if "SSH-Patator" in class_names:
        ssh_idx = class_names.index("SSH-Patator")
        ssh_thr = float(h_cfg.get("ssh_threshold", 0.0))
        ssh_margin = float(h_cfg.get("ssh_margin", 0.0))
        y2 = _apply_ssh_margin_rule(y2, y_proba, ssh_idx, ssh_thr, ssh_margin)

    return y2

def _tune_thresholds_on_val(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    h_cfg: dict,
) -> dict:
    """
    Grid-search thresholds on VAL only, then reuse on TEST (no leakage).
    Objective defaults to macro_f1_excl_ssh, with a weighted-F1 drop guard.
    """
    obj = str(h_cfg.get("threshold_objective", "macro_f1_excl_ssh")).lower().strip()
    wdrop_max = float(h_cfg.get("threshold_weighted_f1_drop_max", 0.01))

    # Baseline (raw argmax) metrics
    m0 = compute_metrics(y_true, y_pred_raw, y_proba, class_names=class_names)
    w0 = float(m0["weighted_f1"])

    # Candidate grids
    ftp_grid = np.linspace(0.20, 0.95, 16)
    ssh_grid = np.linspace(0.20, 0.95, 16)
    ssh_margin_grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]

    best = {
        "ftp_min_threshold": float(h_cfg.get("ftp_min_threshold", 0.0)),
        "ssh_threshold": float(h_cfg.get("ssh_threshold", 0.0)),
        "score": -1e9,
    }

    # Allow “no-op” if user pins thresholds >0
    ftp_pin = float(h_cfg.get("ftp_min_threshold", 0.0)) > 0
    ssh_pin = float(h_cfg.get("ssh_threshold", 0.0)) > 0

    ftp_candidates = [float(h_cfg.get("ftp_min_threshold"))] if ftp_pin else list(ftp_grid) + [0.0]
    ssh_candidates = [float(h_cfg.get("ssh_threshold"))] if ssh_pin else list(ssh_grid) + [0.0]

    # Temporary cfg copy so we can evaluate candidates
    temp = dict(h_cfg)

    for ftp_thr in ftp_candidates:
        temp["ftp_min_threshold"] = float(ftp_thr)
        for ssh_thr in ssh_candidates:
            temp["ssh_threshold"] = float(ssh_thr)
            for ssh_m in ssh_margin_grid:
                temp["ssh_margin"] = float(ssh_m)

                y_pred2 = _apply_decision_rules(y_pred_raw, y_proba, class_names, temp)
                m = compute_metrics(y_true, y_pred2, y_proba, class_names=class_names)

                if ("SSH-Patator" in class_names):
                    cm = np.array(m["confusion_matrix"], dtype=np.int64)
                    m["macro_f1_excl_ssh"] = _macro_f1_excluding_from_cm(cm, list(class_names), "SSH-Patator")

                w = float(m["weighted_f1"])
                if w < (w0 - wdrop_max):
                    continue

                if obj == "macro_f1":
                    score = float(m["macro_f1"])
                else:
                    score = float(m.get("macro_f1_excl_ssh", m["macro_f1"]))

                score2 = score + 0.05 * float(m["accuracy"])

                if score2 > best["score"]:
                    best = {
                        "ftp_min_threshold": float(ftp_thr),
                        "ssh_threshold": float(ssh_thr),
                        "ssh_margin": float(ssh_m),
                        "score": float(score2),
                        "baseline_weighted_f1": w0,
                        "chosen_weighted_f1": w,
                        "chosen_macro_f1": float(m["macro_f1"]),
                        "chosen_macro_f1_excl_ssh": float(m.get("macro_f1_excl_ssh", m["macro_f1"])),
                        "chosen_accuracy": float(m["accuracy"]),
                    }

    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "train", "all"], default="all")
    ap.add_argument("--run_id", default=None)

    # Ablation runner support
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--overrides_json", default=None, help="Path to JSON that deep-merges into configs/base.yaml")
    ap.add_argument("--out_dir", default=None, help="If set, writes artifacts into this exact directory")

    args = ap.parse_args()

    # Determinism + CUDA matmul reproducibility
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    cfg = load_yaml(Path("configs") / "base.yaml").raw

    # Deep-merge overrides (used by ablation runner)
    ovr = _load_json_if_exists(args.overrides_json)
    if ovr:
        _deep_update(cfg, ovr)

    seed_val = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))
    seed_everything(seed_val, deterministic=bool(cfg.get("reproducibility", {}).get("deterministic", True)))

    p4 = cfg["phase4"]
    # Decide whether Seq2Seq pretrain is needed for this run (ablation speed + correctness)
    variant_hint = str(p4.get("hybrid", {}).get("variant", "hybrid")).lower().strip()
    needs_pretrain = variant_hint in ("hybrid", "seq2seq", "seq2seq_only", "s2s_only")
    run_pretrain = (args.stage == "pretrain") or (args.stage == "all" and needs_pretrain)

    out_root = Path(p4.get("output_dir", "results/phase4"))
    run_id = args.run_id or _now_run_id()

    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        run_dir = out_root / "hybrid" / run_id

    _mkdir(run_dir)

    logger = setup_logger("PHASE4_HYBRID", level=str(cfg.get("logging", {}).get("level", "INFO")), log_file=str(run_dir / "run.log"))
    _write_env_snapshot(run_dir)
    (run_dir / "config_used.yaml").write_text((Path("configs") / "base.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    # Persist overrides + resolved seed for reproducibility
    _write_json(run_dir / "overrides_used.json", ovr)
    _write_json(run_dir / "run_seed.json", {"seed": seed_val})

    train = load_seq_npz(p4["seq_npz"]["train"])
    val = load_seq_npz(p4["seq_npz"]["val"])
    test = load_seq_npz(p4["seq_npz"]["test"])
    class_names = train.class_names
    K = len(class_names)
    lt = class_counts_and_priors(train.y, K)
    _write_json(run_dir / "train_class_counts.json", {class_names[i]: int(lt.counts[i]) for i in range(K)})
    _write_json(run_dir / "train_class_priors.json", {class_names[i]: float(lt.priors[i]) for i in range(K)})

    logger.info(f"Train X={train.X.shape} Val X={val.X.shape} Test X={test.X.shape}")
    logger.info(f"Classes={class_names}")

    # Preproc (train-only)
    stats = fit_train_only_stats(train.X)
    Xtr = transform_with_stats(train.X, stats)
    Xva = transform_with_stats(val.X, stats)
    Xte = transform_with_stats(test.X, stats)
    _write_json(run_dir / "preproc_stats.json", stats.to_jsonable())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stage 1: Seq2Seq pretrain ---
    s2s_cfg = p4["seq2seq"]
    ae_path = run_dir / "seq2seq_pretrained.pt"

    if run_pretrain:
        ae = Seq2SeqAutoencoder(
            input_size=Xtr.shape[-1],
            hidden_size=int(s2s_cfg.get("hidden_size", 128)),
            num_layers=int(s2s_cfg.get("num_layers", 1)),
            bidirectional=bool(s2s_cfg.get("bidirectional", True)),
            dropout=float(s2s_cfg.get("dropout", 0.2)),
            cell="gru",
        ).to(device)

        bs = int(s2s_cfg.get("batch_size", 512))
        dl = DataLoader(SeqDS(Xtr), batch_size=bs, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

        opt = torch.optim.AdamW(ae.parameters(), lr=float(s2s_cfg.get("lr", 1e-3)), weight_decay=0.0)
        crit = nn.MSELoss()

        amp = bool(s2s_cfg.get("amp", True)) and (device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        epochs = int(s2s_cfg.get("pretrain_epochs", 4))
        logger.info(f"Seq2Seq pretrain: epochs={epochs} bs={bs} amp={amp}")

        ae.train()
        for ep in range(1, epochs + 1):
            losses = []
            for xb in dl:
                xb = xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    x_hat = ae(xb)
                    loss = crit(x_hat, xb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu().item()))
            logger.info(f"pretrain epoch={ep} mse={float(np.mean(losses)):.6f}")

        torch.save({"ae": ae.state_dict()}, ae_path)
        logger.info(f"Saved pretrained Seq2Seq -> {ae_path.as_posix()}")

    else:
        logger.info(f"Skipping Seq2Seq pretrain for variant={variant_hint}")

    # --- Stage 2: Hybrid classifier training ---
    if args.stage in ("train", "all"):
        h_cfg = p4["hybrid"]
        ssh_idx = class_names.index("SSH-Patator") if "SSH-Patator" in class_names else -1
        variant = str(h_cfg.get("variant", "hybrid")).lower().strip()

        if variant in ("seq2seq", "seq2seq_only", "s2s_only"):
            model = Seq2SeqClassifier(
                input_size=Xtr.shape[-1],
                num_classes=K,
                enc_hidden=int(s2s_cfg.get("hidden_size", 128)),
                enc_layers=int(s2s_cfg.get("num_layers", 1)),
                enc_bidir=bool(s2s_cfg.get("bidirectional", True)),
                enc_dropout=float(s2s_cfg.get("dropout", 0.2)),
                head_dropout=float(h_cfg.get("dropout", 0.2)),
                pooling=str(h_cfg.get("seq2seq_pooling", "mean")),
            ).to(device)

        elif variant in ("convlstm", "convlstm_only", "clstm_only"):
            model = ConvLSTMOnlyClassifier(
                input_size=Xtr.shape[-1],
                num_classes=K,
                convlstm_hidden_channels=int(h_cfg.get("convlstm_hidden_channels", 32)),
                kernels=[int(k) for k in h_cfg.get("kernels", [3, 5, 7])],
                head_dropout=float(h_cfg.get("dropout", 0.2)),
                feature_map_mode=str(h_cfg.get("feature_map_mode", "none")),
                map_h=int(h_cfg.get("map_h", 1)),
                map_w=int(h_cfg.get("map_w", 1)),
            ).to(device)

        else:
            model = HybridSeq2SeqConvLSTM(
                input_size=Xtr.shape[-1],
                num_classes=K,
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

        if ae_path.exists():
            ck = torch.load(ae_path, map_location=device)
            if hasattr(model, "ae") and hasattr(model.ae, "load_state_dict"):
                model.ae.load_state_dict(ck["ae"], strict=False)
                logger.info("Loaded pretrained Seq2Seq weights into model.ae.")

        bs = int(h_cfg.get("batch_size", 512))
        train_sampler = str(h_cfg.get("train_sampler", "none")).lower().strip()
        if train_sampler == "weighted":
            sampler = _make_weighted_sampler(train.y)
            dl_tr = DataLoader(
                SeqDS(Xtr, train.y),
                batch_size=bs,
                sampler=sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
        else:
            dl_tr = DataLoader(
                SeqDS(Xtr, train.y),
                batch_size=bs,
                shuffle=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

        dl_va = DataLoader(SeqDS(Xva, val.y), batch_size=bs, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
        dl_te = DataLoader(SeqDS(Xte, test.y), batch_size=bs, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

        cap = float(h_cfg.get("class_weight_cap", 20.0))
        train_sampler = str(h_cfg.get("train_sampler", "none")).lower().strip()

        loss_name = str(h_cfg.get("loss", "ce")).lower().strip()
        label_smooth = float(h_cfg.get("label_smoothing", 0.0))
        focal_gamma = float(h_cfg.get("focal_gamma", 2.0))

        # Long-tail knobs (train-only)
        la_tau = float(h_cfg.get("la_tau", 1.0))
        la_eps = float(h_cfg.get("la_eps", 1e-12))

        if train_sampler == "weighted":
            w = None  # sampler already balances; don't double-weight
        else:
            w = _class_weights(train.y, K, cap=cap, device=device)

        if loss_name in ("la_ce", "logit_adjusted", "logit_adjusted_ce"):
            # Train-only: loss uses adjusted logits; evaluation still uses raw logits.
            crit = LogitAdjustedCrossEntropy(lt.priors, tau=la_tau, label_smoothing=label_smooth, eps=la_eps)
            logger.info(f"loss=la_ce tau={la_tau:.4g} label_smoothing={label_smooth:.4g}")

        elif loss_name in ("balanced_softmax", "bs"):
            crit = BalancedSoftmaxLoss(lt.counts, label_smoothing=label_smooth, eps=la_eps)
            logger.info(f"loss=balanced_softmax label_smoothing={label_smooth:.4g}")

        elif loss_name in ("label_smoothing", "ls"):
            crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smooth)

        elif loss_name in ("focal", "focal_loss"):
            crit = FocalLoss(gamma=focal_gamma, weight=w, label_smoothing=label_smooth)

        else:
            crit = nn.CrossEntropyLoss(weight=w)

        ssh_aux = bool(h_cfg.get("ssh_aux", False))
        ssh_aux_w = float(h_cfg.get("ssh_aux_weight", 0.35))
        ssh_pos_cap = float(h_cfg.get("ssh_pos_weight_cap", 50.0))

        bce_ssh = None
        if ssh_aux and ("SSH-Patator" in class_names):
            ssh_idx = class_names.index("SSH-Patator")
            n_pos = max(1, int((train.y == ssh_idx).sum()))
            n_neg = int(len(train.y) - n_pos)
            pos_w = min(ssh_pos_cap, float(n_neg) / float(n_pos))
            bce_ssh = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
            logger.info(f"ssh_aux=on pos_weight={pos_w:.3g} aux_weight={ssh_aux_w:.3g}")

        base_lr = float(h_cfg.get("lr", 1e-3))
        enc_mult = float(h_cfg.get("encoder_lr_mult", 0.1))

        # Two-phase fine-tune controls
        head_lr_unf_mult = float(h_cfg.get("head_lr_mult_unfrozen", 1.0))
        enc_lr_unf_mult = float(h_cfg.get("encoder_lr_mult_unfrozen", enc_mult))
        sched_name = str(h_cfg.get("scheduler", "none")).lower().strip()
        sched_after_unfreeze_only = bool(h_cfg.get("scheduler_after_unfreeze_only", True))

        onecycle_pct_start = float(h_cfg.get("onecycle_pct_start", 0.2))
        onecycle_div_factor = float(h_cfg.get("onecycle_div_factor", 10.0))
        onecycle_final_div_factor = float(h_cfg.get("onecycle_final_div_factor", 200.0))

        eta_min = float(h_cfg.get("cosine_eta_min", 1e-6))

        enc_lr_init = base_lr * enc_mult
        head_lr_init = base_lr

        enc_lr_unf = base_lr * enc_lr_unf_mult
        head_lr_unf = base_lr * head_lr_unf_mult

        has_encoder = hasattr(model, "ae") and hasattr(model.ae, "encoder")
        if not has_encoder:
            freeze_epochs = 0

        if has_encoder:
            opt = torch.optim.AdamW(
                [
                    {"params": model.ae.encoder.parameters(), "lr": enc_lr_init},
                    {"params": [p for n, p in model.named_parameters() if not n.startswith("ae.encoder")], "lr": head_lr_init},
                ],
                weight_decay=float(h_cfg.get("weight_decay", 0.0)),
            )
        else:
            opt = torch.optim.AdamW(
                [{"params": model.parameters(), "lr": head_lr_init}],
                weight_decay=float(h_cfg.get("weight_decay", 0.0)),
            )

        def _set_lrs(enc_lr: float, head_lr: float) -> None:
            if has_encoder and len(opt.param_groups) >= 2:
                opt.param_groups[0]["lr"] = float(enc_lr)
                opt.param_groups[1]["lr"] = float(head_lr)
            else:
                # Single-group optimizer (ConvLSTM-only): treat head_lr as the only lr.
                opt.param_groups[0]["lr"] = float(head_lr)

        def _build_scheduler(total_steps: int, max_lr_enc: float, max_lr_head: float):
            if sched_name == "onecycle":
                max_lr = float(max_lr_head) if len(opt.param_groups) == 1 else [float(max_lr_enc), float(max_lr_head)]
                return torch.optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=max_lr,
                    total_steps=int(total_steps),
                    pct_start=onecycle_pct_start,
                    div_factor=onecycle_div_factor,
                    final_div_factor=onecycle_final_div_factor,
                    anneal_strategy="cos",
                )
            if sched_name == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(total_steps), eta_min=eta_min)
            return None

        sched = None  # created either immediately or at unfreeze boundary

        amp = bool(h_cfg.get("amp", True)) and (device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        freeze_epochs = int(h_cfg.get("freeze_encoder_epochs", 0))
        if not has_encoder:
            freeze_epochs = 0

        max_gn = float(h_cfg.get("max_grad_norm", 1.0))

        epochs = int(h_cfg.get("epochs", 15))
        patience = int(h_cfg.get("patience", 4))

        best = -1.0
        bad = 0

        best_path = run_dir / "best.pt"
        last_path = run_dir / "last.pt"

        for ep in range(1, epochs + 1):

            # Phase A (frozen) uses encoder lr=0 to avoid any schedule side-effects.
            if freeze_epochs > 0 and ep == 1:
                _set_lrs(0.0, head_lr_init)

            # Create scheduler immediately only if we are NOT delaying until after unfreeze.
            if ep == 1 and sched_name != "none" and not (sched_after_unfreeze_only and freeze_epochs > 0):
                total_steps = epochs * len(dl_tr)
                if sched_name == "onecycle":
                    sched = _build_scheduler(total_steps, enc_lr_unf, head_lr_unf)
                else:
                    sched = _build_scheduler(total_steps, enc_lr_unf, head_lr_unf)
                logger.info(f"scheduler={sched_name} steps={total_steps} after_unfreeze_only={False}")

            # Freeze encoder early to prevent representation drift
            freeze = (freeze_epochs > 0) and (ep <= freeze_epochs)

            if has_encoder:
                for p in model.ae.encoder.parameters():
                    p.requires_grad = (not freeze)
                if freeze:
                    logger.info(f"epoch={ep}: encoder frozen")

            # Unfreeze boundary: set fine-tune LRs and (optionally) start scheduler now.
            if has_encoder and freeze_epochs > 0 and ep == (freeze_epochs + 1):
                _set_lrs(enc_lr_unf, head_lr_unf)
                if sched_name != "none" and sched_after_unfreeze_only:
                    total_steps = (epochs - freeze_epochs) * len(dl_tr)
                    if sched_name == "onecycle":
                        sched = _build_scheduler(total_steps, enc_lr_unf, head_lr_unf)
                        logger.info(
                            f"scheduler=onecycle steps={total_steps} after_unfreeze_only=True "
                            f"max_lr_enc={enc_lr_unf:.3g} max_lr_head={head_lr_unf:.3g}"
                        )
                    else:
                        _set_lrs(enc_lr_unf, head_lr_unf)
                        sched = _build_scheduler(total_steps, enc_lr_unf, head_lr_unf)
                        logger.info(f"scheduler={sched_name} steps={total_steps} after_unfreeze_only=True")

                    logger.info(
                        f"epoch={ep}: encoder unfrozen enc_lr_init={opt.param_groups[0]['lr']:.3g} "
                        f"head_lr_init={opt.param_groups[min(1, len(opt.param_groups)-1)]['lr']:.3g}"
                    )

            model.train()
            losses = []
            for xb, yb in dl_tr:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    logits = model(xb)
                    loss_main = crit(logits, yb)
                    loss = loss_main

                    if bce_ssh is not None:
                        # Recompute the gate logit effect indirectly by reading the SSH logit “excess”
                        # We train the model to raise the SSH logit for true SSH.
                        y_ssh = (yb == ssh_idx).float()
                        ssh_logit = logits[:, ssh_idx]
                        loss_aux = bce_ssh(ssh_logit, y_ssh)
                        loss = loss_main + (ssh_aux_w * loss_aux)

                scaler.scale(loss).backward()
                if max_gn > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
                scaler.step(opt)
                scaler.update()
                if sched is not None:
                    sched.step()
                losses.append(float(loss.detach().cpu().item()))

            # val eval
            y_true, y_pred, y_proba = _predict(model, dl_va, device, K)

            # Apply unified decision rules (FTP demotion + SSH margin rule)
            y_pred_eval = _apply_decision_rules(y_pred, y_proba, class_names, h_cfg)

            m = compute_metrics(y_true, y_pred_eval, y_proba, class_names=class_names)

            if bool(h_cfg.get("report_macro_f1_excl_ssh", True)) and ("SSH-Patator" in class_names):
                cm = np.array(m["confusion_matrix"], dtype=np.int64)
                m["macro_f1_excl_ssh"] = _macro_f1_excluding_from_cm(cm, list(class_names), "SSH-Patator")

            if "macro_f1_excl_ssh" in m:
                logger.info(
                    f"epoch={ep} loss={float(np.mean(losses)):.6f} "
                    f"val_macro_f1={m['macro_f1']:.6f} val_macro_f1_excl_ssh={m['macro_f1_excl_ssh']:.6f} "
                    f"val_weighted_f1={m['weighted_f1']:.6f}"
                )
            else:
                logger.info(f"epoch={ep} loss={float(np.mean(losses)):.6f} val_macro_f1={m['macro_f1']:.6f} val_weighted_f1={m['weighted_f1']:.6f}")

            torch.save({"model": model.state_dict(), "epoch": ep, "val_macro_f1": m["macro_f1"]}, last_path)

            es_key = str(h_cfg.get("early_stop_metric", "macro_f1")).lower().strip()
            score = float(m.get(es_key, m["macro_f1"]))
            if score > best:
                best = score
                bad = 0
                torch.save({"model": model.state_dict(), "epoch": ep, "val_macro_f1": m["macro_f1"]}, best_path)
            else:
                bad += 1
                if bad >= patience:
                    logger.info(f"Early stopping: patience={patience} reached.")
                    break

        # load best and evaluate val/test + save artifacts
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model"])

        def _eval_save(split_name: str, dl: DataLoader) -> Dict[str, Any]:
            y_true, y_pred, y_proba = _predict(model, dl, device, K)

            # Apply FTP/SSH decision rules (thresholds possibly tuned on VAL)
            y_pred2 = _apply_decision_rules(y_pred, y_proba, class_names, h_cfg)

            metrics = compute_metrics(y_true, y_pred2, y_proba, class_names=class_names)

            if bool(h_cfg.get("report_macro_f1_excl_ssh", True)) and ("SSH-Patator" in class_names):
                cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
                metrics["macro_f1_excl_ssh"] = _macro_f1_excluding_from_cm(cm, list(class_names), "SSH-Patator")

            # Save BOTH raw and post-rule predictions; also avoid object dtype for class_names
            np.savez_compressed(
                run_dir / f"pred_{split_name}.npz",
                y_true=y_true.astype(np.int64),
                y_pred_raw=y_pred.astype(np.int64),
                y_pred=y_pred2.astype(np.int64),
                y_proba=y_proba.astype(np.float32),
                class_names=np.array(class_names, dtype="U32"),
            )

            cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
            # Per-class evidence (publishable, stable)
            metrics["per_class"] = _per_class_prf_from_cm(cm, list(class_names))

            # Explicit SSH metrics (separate headline number; avoids macro volatility confusion)
            if "SSH-Patator" in class_names:
                sidx = class_names.index("SSH-Patator")
                ssh = metrics["per_class"].get("SSH-Patator", {})
                metrics["ssh_precision"] = float(ssh.get("precision", float("nan")))
                metrics["ssh_recall"] = float(ssh.get("recall", float("nan")))
                metrics["ssh_f1"] = float(ssh.get("f1", float("nan")))
                metrics["ssh_support"] = int(ssh.get("support", 0))

            save_confusion_csv(str(run_dir / f"confusion_{split_name}.csv"), cm, class_names, normalize=False)
            save_confusion_csv(str(run_dir / f"confusion_{split_name}_norm.csv"), cm, class_names, normalize=True)
            return metrics

        if bool(h_cfg.get("auto_tune_thresholds", False)):
            yv, ypv, pv = _predict(model, dl_va, device, K)
            best = _tune_thresholds_on_val(yv, ypv, pv, class_names, h_cfg)

            # Write chosen thresholds back into h_cfg for consistent evaluation
            h_cfg["ftp_min_threshold"] = float(best["ftp_min_threshold"])
            h_cfg["ssh_threshold"] = float(best["ssh_threshold"])
            h_cfg["ssh_margin"] = float(best.get("ssh_margin", 0.0))

            logger.info(
                "VAL threshold-tune: ftp_min_threshold=%.3f ssh_threshold=%.3f "
                "val_macro_f1=%.6f val_macro_f1_excl_ssh=%.6f val_weighted_f1=%.6f val_acc=%.6f",
                h_cfg["ftp_min_threshold"], h_cfg["ssh_threshold"],
                best.get("chosen_macro_f1", float("nan")),
                best.get("chosen_macro_f1_excl_ssh", float("nan")),
                best.get("chosen_weighted_f1", float("nan")),
                best.get("chosen_accuracy", float("nan")),
            )

            # Persist for reproducibility
            with open(run_dir / "thresholds.json", "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2)

        m_val = _eval_save("val", dl_va)
        m_test = _eval_save("test", dl_te)

        out = {"val": m_val, "test": m_test}
        _write_json(run_dir / "metrics.json", out)

        import pandas as pd
        rows = metrics_to_flat_rows("val", m_val) + metrics_to_flat_rows("test", m_test)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)

        # Always persist macro_f1_excl_ssh (if present) in a stable schema.
        extra_rows = []
        for split, mm in [("val", m_val), ("test", m_test)]:
            if "macro_f1_excl_ssh" in mm:
                extra_rows.append({"split": split, "metric": "macro_f1_excl_ssh", "value": mm["macro_f1_excl_ssh"]})
        if extra_rows:
            pd.DataFrame(extra_rows).to_csv(run_dir / "metrics_extra.csv", index=False)

        logger.info(f"Wrote run_dir: {run_dir.as_posix()}")


if __name__ == "__main__":
    main()
