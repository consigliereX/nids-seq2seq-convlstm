# CICIDS2017 Seq2Seq–ConvLSTM NIDS

A research-to-deployment repository for network intrusion detection on the **CICIDS2017** dataset using:

- classical baselines (**Random Forest** and **RNN**)
- a **Seq2Seq + ConvLSTM-style hybrid** pipeline
- a reproducible **preprocessing → sequence building → training → inference** workflow
- a packaged **artifact bundle** for batch inference

This repository is suitable for:

- final-year / capstone academic work
- IDS research benchmarking
- portfolio demonstration of sequence-based intrusion detection
- extending into deployment or further model experimentation

---

## Overview

This project starts from raw CICIDS2017 CSV files, removes leakage-prone fields, preserves source-file provenance, builds fixed-length sequences, and uses train-only preprocessing statistics to reduce leakage.

The repository is organized around three major goals:

1. **Reproducible preprocessing**
2. **Baseline comparison**
3. **Hybrid model inference from saved artifacts**

---

## What this repository contains

- raw-to-processed data preparation
- fixed-length sequence generation
- phase-wise experimentation
- Random Forest and RNN baselines
- Seq2Seq + ConvLSTM-style hybrid training path
- saved artifact bundle for the selected hybrid model
- batch inference entrypoint for `.npz` sequence inputs
- tests for metrics and preprocessing utilities

---

## Repository structure

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

> `outputs/` is generated after inference or experiments and does not need to exist before you run the pipeline.

---

## Dataset expectation

The configuration expects raw CICIDS2017 data under a layout like:

```text
data/raw/CICIDS2017/
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

The preprocessing config removes leakage-prone fields such as IPs, ports, timestamps, and flow identifiers before training.

---

## Pipeline

```text
Raw CICIDS2017 CSVs
    ↓
src/preprocess_cicids2017.py
    ↓
cleaned parquet / processed data
    ↓
src/make_sequences.py
    ↓
sequence NPZ files
    ↓
Phase 3 baselines / Phase 4 hybrid training
    ↓
saved artifacts
    ↓
app/infer.py
```

---

## Reported results

### Baseline results

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|------|---------:|------------:|---------:|
| Random Forest | 0.729681 | 0.998237 | 0.998400 |
| RNN | 0.707840 | 0.996966 | 0.996312 |

### Selected hybrid artifact bundle

The repository includes a saved hybrid model bundle in:

```text
artifacts/CICIDS2017_best_hybrid/
```

Important files:

- `best.pt`
- `config_used.yaml`
- `preproc_stats.json`
- `thresholds.json`
- `metrics.json`

### Hybrid metrics from shipped artifacts

| Split | Macro-F1 | Weighted-F1 | Accuracy |
|------|---------:|------------:|---------:|
| Validation | 0.608523 | 0.990252 | 0.985979 |
| Test | 0.601309 | 0.989140 | 0.984171 |

> Note: ultra-rare classes such as `SSH-Patator` heavily affect macro metrics. The shipped artifact bundle also reports `macro_f1_excl_ssh` for a more stable reading of general multiclass performance.

---

## Installation

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Main entrypoints

Run the pipeline stages with:

```bash
python -m src.preprocess_cicids2017
python -m src.make_sequences
python -m src.phase3_baselines
python -m src.phase4_hybrid
python -m src.phase4_ablation
```

---

## Batch inference

The repository includes a working batch inference script:

```bash
python app/infer.py \
  --input data/processed/sequences/cicids2017_seq_T20_S5_test.npz \
  --artifacts-dir artifacts/CICIDS2017_best_hybrid \
  --output-dir outputs/test_infer
```

### Generated outputs

After inference, the script writes:

- `outputs/test_infer/predictions.csv`
- `outputs/test_infer/summary.json`
- `outputs/test_infer/metrics.json` *(when `y` is present in the input NPZ)*

---

## What `app/infer.py` expects

`app/infer.py` is designed for **sequence NPZ inference**, not direct raw-CSV inference.

### Required input

- `X` with shape `(N, T, d)`

### Optional input

- `y`
- `class_names`
- `feature_names`

If `class_names` are not provided in the input NPZ, the script expects them to be recoverable from `pred_test.npz` or `pred_val.npz` inside the artifact directory.

---

## Testing

Run the test suite with:

```bash
pytest -q
```

Included tests:

- `tests/test_metrics_fpr.py`
- `tests/test_seq_preproc_shapes.py`

---

## Why this repo is useful

This is not just a notebook or a single training script. It already includes:

- preprocessing logic
- leakage-aware train-only statistics
- baseline comparison
- hybrid experimentation
- saved model artifacts
- threshold tuning outputs
- inference scripts
- tests

That makes it a strong starting point for:

- academic reporting
- benchmark comparisons
- model extension and ablation work
- conversion into a deployment-facing IDS workflow

---

## Limitations

- Performance on ultra-rare classes remains challenging.
- Inference expects prebuilt sequence NPZ files rather than raw network flows.
- The repository is best treated as a **research-grade batch pipeline**, not a production IDS appliance out of the box.

---

## Future improvements

- richer experiment tracking
- direct raw-flow inference support
- stronger handling of rare attack classes
- deployment wrapper / API layer
- model explainability and calibration analysis

---

## License

The original source code is distributed under the [MIT License](LICENSE).

Third-party datasets, pretrained model artifacts, and externally sourced files may have separate licenses or redistribution terms. Check their provenance and permissions before reuse; this repository's MIT notice does not supersede third-party rights.
