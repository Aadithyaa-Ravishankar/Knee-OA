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
Knee-OA/
├── docs/PIPELINE_STAGES.md      # Canonical Stage 1–11 numbering
├── src/
│   ├── preprocessing/           # Stage 1–2 (+ fold feature builders)
│   ├── models/                  # OrdinalNet dual-stream (champion)
│   ├── losses/                  # Ordinal / cascade losses
│   ├── evaluation/              # Metrics, decoding, Grad-CAM, ESS, ECE
│   └── training/                # stage4–11 CV / decode / XAI / UQ scripts
├── outputs/cv/runs/             # 5-fold champion weights
├── reports/pipeline/            # Folds, finals, Grad-CAM, consistency, UQ
├── kaggle/                      # Train notebook + code/data bundles
├── Dataset_combined/            # Local data (not in git)
├── requirements.txt
├── DATASET_SETUP.md
└── README.md
```

See **[docs/PIPELINE_STAGES.md](docs/PIPELINE_STAGES.md)** for the full Stage 1–11 map.

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
python src/preprocessing/stage1_dataset_audit.py
```

### 5. Locked-test style inference (Stages 6–7)
```bash
python src/training/stage7_final_test.py \
  --run-name r50_dual_cascade_sqrt_aux_attn_spatial \
  --runs-dir outputs/cv/runs \
  --decode expected_cutpoints \
  --i-understand-this-is-the-final-test
```

### 6. Grad-CAM proof on correct grades (Stage 8)
```bash
python src/training/stage8_gradcam.py \
  --run-name r50_dual_cascade_sqrt_aux_attn_spatial \
  --runs-dir outputs/cv/runs \
  --split test --errors correct --decode expected_cutpoints --n 40 --cam-folds 0 \
  --out reports/pipeline/gradcam_correct_attn_spatial_cutpoints
```

### 7. Consistency / uncertainty / ablation (Stages 9–11)
```bash
python src/training/stage9_explanation_consistency.py --n 200 --decode expected_cutpoints
python src/training/stage10_uncertainty.py --n-passes 30 --decode expected_cutpoints
python src/training/stage11_ablation_report.py
```

---

## Implementation Stages

| Stage | Status | Description |
|------:|--------|-------------|
| 1 | ✅ | Dataset audit |
| 2 | ✅ | Preprocessing (CLAHE, normalize, loaders) |
| 3 | ✅ | Early dual-cascade / baseline era |
| 4 | ✅ | Patient-grouped 5-fold CV + OOF |
| 5 | ✅ | OARSI aux (JSN/ost) multi-task |
| 6 | ✅ | Joint-focus champion (`attn_spatial`) |
| 7 | ✅ | Expected-grade cutpoints + locked test (**QWK 0.867**) |
| 8 | ✅ | Grad-CAM explainability |
| 9 | ✅ | Explanation consistency (AFS / ESS / grade flip) |
| 10 | ✅ | Uncertainty & calibration (MC Dropout / ECE) |
| 11 | ✅ | Ablation reporting (artifact table) |

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

---

## Champion model (Stages 6–7)

| Item | Value |
|------|-------|
| Weights | `outputs/cv/runs/r50_dual_cascade_sqrt_aux_attn_spatial/` (5-fold) |
| Decode | Expected-grade cutpoints (fit on OOF) |
| Locked-test QWK | **0.8673** |
| Report | `reports/pipeline/final_test_r50_dual_cascade_sqrt_aux_attn_spatial_expected_cutpoints.json` |

See `docs/PIPELINE_STAGES.md` and `RELEASE_MANIFEST.md`.
