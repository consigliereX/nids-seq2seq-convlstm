# CICIDS2017 Seq2Seq–ConvLSTM NIDS

A research-to-deployment repository for **network intrusion detection on CICIDS2017** using:
- classical baselines (**Random Forest**, **RNN**)
- a **Phase 4 hybrid path** built around a **Seq2Seq encoder + ConvLSTM-style classifier**
- a working **batch inference entrypoint** for sequence-based evaluation

## Overview

This project starts from raw CICIDS2017 CSVs, preserves provenance, removes leakage-prone fields, builds fixed-length sequences, and uses **time-aware train/validation/test splits**. Imputation and scaling statistics are computed on **train only** to reduce leakage risk.

The repository now separates:
- active code in `src/`
- selected deployment artifacts in `artifacts/CICIDS2017_best_hybrid/`
- historical experiments in `archive/`
- batch inference in `app/infer.py`

## Why this repo is useful

This is not just a training script dump. It already contains:
- preprocessing and sequence construction
- baseline comparisons
- hybrid training and ablations
- saved checkpoints
- threshold tuning artifacts
- confusion matrices and metrics
- a tested inference path that produces prediction outputs

That makes it suitable for:
- a final-year project repository
- a portfolio-grade IDS research repo
- a benchmark foundation for future experiments
- a deployable batch inference demo

## Methodology summary

The pipeline is designed to be leakage-aware and reproducible:

- starts from raw CICIDS2017 CSVs
- preserves provenance through preprocessing
- removes leakage-prone fields
- builds fixed-length sequences
- uses time-aware splitting
- fits imputation/scaling on train only
- keeps artifacts from preprocessing to final metrics

The architecture direction behind the project title is already present:
- **RF baseline**
- **RNN baseline**
- **Seq2Seq + ConvLSTM hybrid path**

## Reported baseline results

From the project summary:

### Random Forest
- Macro-F1: **0.729681**
- Weighted-F1: **0.998237**
- Accuracy: **0.998400**

### RNN
- Macro-F1: **0.707840**
- Weighted-F1: **0.996966**
- Accuracy: **0.996312**

These are strong reference baselines for the Phase 4 hybrid line.

## Current repository structure

```text
nids-seq2seq-convlstm/
├── app/
│   └── infer.py
├── archive/
│   ├── phase3/
│   └── phase4/
├── artifacts/
│   └── CICIDS2017_best_hybrid/
├── configs/
│   └── base.yaml
├── data/
│   └── processed/
│       └── sequences/
├── outputs/
│   └── test_infer/
├── scripts/
│   └── run_env_check.ps1
├── src/
│   ├── cli_env_check.py
│   ├── make_sequences.py
│   ├── phase3_baselines.py
│   ├── phase4_ablation.py
│   ├── phase4_hybrid.py
│   ├── preprocess_cicids2017.py
│   ├── models/
│   └── utils/
├── tests/
│   ├── test_metrics_fpr.py
│   └── test_seq_preproc_shapes.py
├── README.md
└── requirements.txt
```

## Selected artifact bundle

The selected hybrid model bundle is stored in:

```text
artifacts/CICIDS2017_best_hybrid/
```

Important files inside it:
- `best.pt`
- `config_used.yaml`
- `preproc_stats.json`
- `thresholds.json`
- `metrics.json`

Additional files such as `last.pt`, `pred_test.npz`, and logs are useful for research traceability but are not all required for deployment.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Inference

The repository now includes a working batch inference command:

```bash
python app/infer.py   --input data/processed/sequences/cicids2017_seq_T20_S5_test.npz   --artifacts-dir artifacts/CICIDS2017_best_hybrid   --output-dir outputs/test_infer
```

Verified outputs:
- `outputs/test_infer/predictions.csv`
- `outputs/test_infer/summary.json`
- `outputs/test_infer/metrics.json`

## What `infer.py` expects

`app/infer.py` is designed for **sequence NPZ inference**, not raw CSV ingestion.

Expected input NPZ:
- `X`: shape `(N, T, d)`

Optional:
- `y`
- `class_names`
- `feature_names`

Recommended flow:

```text
raw CICIDS CSVs
→ src/preprocess_cicids2017.py
→ src/make_sequences.py
→ sequence NPZ
→ app/infer.py
```

## Training / experimentation entrypoints

Main code paths:

```bash
python -m src.preprocess_cicids2017
python -m src.make_sequences
python -m src.phase3_baselines
python -m src.phase4_hybrid
python -m src.phase4_ablation
```

## Tests

Run tests with:

```bash
pytest -q
```

Included tests:
- `tests/test_metrics_fpr.py`
- `tests/test_seq_preproc_shapes.py`

## Recommended cleanup choices

### Keep if you want a research repo
- `archive/`
- confusion matrices
- older metrics
- ablation summaries

### Trim if you want a deployment-first repo
- old seed folders in `archive/`
- extra logs
- `last.pt`
- `pred_test.npz`
- `pred_val.npz`
- extra artifact files not needed for inference

## Recommended next steps

1. Remove `__pycache__` folders before pushing.
2. Use the included `.gitignore` to ignore `outputs/`, virtual environments, and cache files.
3. Decide whether `archive/` stays in the public repo or moves to separate storage.
4. Optionally slim `artifacts/CICIDS2017_best_hybrid/` to deployment-only files.
5. Add architecture diagrams or confusion matrix images later for stronger presentation.

## Current status

The repository now has:
- a cleaned top-level layout
- isolated selected hybrid artifacts
- historical experiment archives
- a working sequence-based inference command
- generated inference outputs

That makes it a valid GitHub-ready IDS project with both research depth and deployment direction.
