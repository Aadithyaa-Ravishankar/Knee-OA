# Anatomy-Aware Deep Learning with Consistent Explainable AI for Knee OA Severity Classification

> An Explainable AI Approach for Automated Knee X-ray Severity Assessment using the Kellgren–Lawrence (KL) grading system.

**Authors:** Aadithyaa R · Mohanapriya V · Sneha S R  
**Institution:** SRM Institute of Science and Technology

---

## Overview

This project implements a full research pipeline for automated Knee Osteoarthritis (OA) severity grading from plain X-rays. The system grades knee X-rays on the Kellgren–Lawrence (KL) scale from **KL-0 (Normal)** to **KL-4 (Severe)**.

The four key pillars of the framework:

| Pillar | What it does |
|--------|-------------|
| **Anatomy-Aware Learning** | Dual-stream ResNet-50 with global + local knee-region focus |
| **Ordinal Severity Learning** | Loss function that respects KL-0 < KL-1 < … < KL-4 ordering |
| **Explainable AI (Grad-CAM)** | Heatmap showing which region of the X-ray drove the prediction |
| **Explanation Reliability** | Perturbation analysis + uncertainty quantification |

---

## Dataset Setup

> ⚠️ **The dataset is NOT included in this repository** (too large for GitHub).  
> See **[DATASET_SETUP.md](DATASET_SETUP.md)** for step-by-step download and placement instructions.

---

## Project Structure

```
knee_oa_xai/
│
├── configs/
│   └── config.yaml              # Central config (paths, hyperparams)
│
├── data/
│   └── raw/                     # Placeholder — put dataset HERE (see DATASET_SETUP.md)
│
├── src/
│   ├── preprocessing/
│   │   ├── dataset.py           # PyTorch Dataset class
│   │   ├── transforms.py        # CLAHE + Normalize transforms
│   │   ├── dataloader.py        # Weighted DataLoader factory
│   │   ├── compute_stats.py     # Compute mean/std from training split
│   │   └── stage1_dataset_audit.py   # Stage 1 audit script
│   │
│   ├── models/
│   │   └── baseline.py          # ResNet-50 baseline (Stage 3)
│   │
│   ├── evaluation/
│   │   └── metrics.py           # QWK, Macro-F1, confusion matrix
│   │
│   ├── training/
│   │   └── stage3_train_baseline.py  # Stage 3 training script
│   │
│   ├── localization/            # Stage TBD — knee region localization
│   ├── losses/                  # Stage TBD — ordinal loss functions
│   ├── xai/                     # Stage TBD — Grad-CAM
│   └── uncertainty/             # Stage TBD — MC Dropout
│
├── outputs/
│   ├── figures/                 # Dataset visualizations
│   ├── metrics/                 # Audit CSVs
│   ├── models/                  # Saved model checkpoints (.pth)
│   └── heatmaps/                # Grad-CAM outputs (Stage TBD)
│
├── reports/
│   ├── stage1/                  # Dataset audit report + figures
│   ├── stage2/                  # Normalization stats + preprocessing report
│   └── stage3/                  # Training curves + confusion matrices
│
├── requirements.txt
├── DATASET_SETUP.md             # ← READ THIS FIRST
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Aadithyaa-Ravishankar/Knee-OA.git
cd Knee-OA
```

### 2. Download the dataset
Follow **[DATASET_SETUP.md](DATASET_SETUP.md)** exactly.

### 3. Set up the environment
```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 4. Run the dataset audit (Stage 1)
```bash
python3 src/preprocessing/stage1_dataset_audit.py
# Output → reports/stage1/
```

### 5. Compute normalization stats (Stage 2)
```bash
python3 src/preprocessing/compute_stats.py
# Output → reports/stage2/normalization_stats.json
```

### 6. Train the baseline model (Stage 3)
```bash
python3 src/training/stage3_train_baseline.py
# Output → outputs/models/baseline_resnet50_best.pth
#           reports/stage3/
```

---

## Implementation Stages

| Stage | Status | Description |
|-------|--------|-------------|
| Stage 1 | ✅ Complete | Dataset audit — 9,786 images, KL-0..4, no leakage |
| Stage 2 | ✅ Complete | Preprocessing pipeline — CLAHE, normalize, weighted sampler |
| Stage 3 | 🔄 In Progress | Baseline ResNet-50 training + evaluation |
| Stage 4 | ⬜ Planned | Dual-stream architecture + ordinal loss |
| Stage 5 | ⬜ Planned | Grad-CAM explainability |
| Stage 6 | ⬜ Planned | Explanation consistency + uncertainty quantification |

---

## Key Results (Stage 1 Audit)

| Property | Finding |
|----------|---------|
| Total images | 9,786 |
| Format | PNG, Grayscale, 224×224 px |
| Corrupted images | 0 |
| Duplicates | 0 |
| Class imbalance | 13.07× (KL-0 most, KL-4 fewest) |
| Patient IDs | Parseable from filenames |
| Data leakage | None detected |

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.1
- torchvision ≥ 0.16
- opencv-python ≥ 4.8
- scikit-learn ≥ 1.3
- See `requirements.txt` for complete list

---

## Acknowledgements

Dataset: [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) — Kaggle

---

## License

For academic and research use only.
