# Release manifest — Knee-OA champion pipeline (Stages 6–11)

Canonical stage numbering: **[docs/PIPELINE_STAGES.md](docs/PIPELINE_STAGES.md)**.  
Retired label: experimental “Stage 19” (renamed into Stages 4–8).

## Champion path

| Piece | Location |
|-------|----------|
| Weights (5 folds) | `outputs/cv/runs/r50_dual_cascade_sqrt_aux_attn_spatial/` |
| Locked test (cutpoints) | `reports/pipeline/final_test_r50_dual_cascade_sqrt_aux_attn_spatial_expected_cutpoints.json` |
| Locked test (native ref) | `reports/pipeline/final_test_r50_dual_cascade_sqrt_aux_attn_spatial.json` |
| Grad-CAM proof (correct) | `reports/pipeline/gradcam_correct_attn_spatial_cutpoints/` |
| Explanation consistency | `reports/pipeline/explanation_consistency_r50_dual_cascade_sqrt_aux_attn_spatial/` |
| Uncertainty / ECE | `reports/pipeline/uncertainty_r50_dual_cascade_sqrt_aux_attn_spatial/` |
| Ablation table | `reports/pipeline/ablation_stage11/` |
| Folds / aux labels | `reports/pipeline/folds.csv`, `folds_features.csv` |

**Locked-test QWK (final decode): 0.8673**

## Core code

- Models: `src/models/ordinal_net.py` (+ legacy `dual_stream*.py` for Stage 3 history)
- Losses: `src/losses/`
- Evaluation: `src/evaluation/` (decoding, Grad-CAM dual, consistency, uncertainty, protocol)
- Training: `stage4_*` … `stage11_*` (CV / aux / screen / decode / Grad-CAM / ESS / MC-ECE / ablation)
- Preprocessing: `src/preprocessing/`

## Kaggle

- Notebook: `kaggle/knee_oa_cv_train.ipynb`
- Bundle: `kaggle/make_bundle.py` → `knee-oa-code.zip` / `knee-oa-data.zip`
- Working runs dir name: `cv_runs/`; results zip: `results_knee_oa.zip`

## Quick use

```bash
# Locked-test style decode (cutpoints) — refuse if report already exists
python src/training/stage7_final_test.py \
  --run-name r50_dual_cascade_sqrt_aux_attn_spatial \
  --runs-dir outputs/cv/runs \
  --decode expected_cutpoints \
  --i-understand-this-is-the-final-test

# Grad-CAM on correctly graded test images
python src/training/stage8_gradcam.py \
  --run-name r50_dual_cascade_sqrt_aux_attn_spatial \
  --runs-dir outputs/cv/runs \
  --errors correct --decode expected_cutpoints --n 40 --cam-folds 0 \
  --out reports/pipeline/gradcam_correct_attn_spatial_cutpoints

# Stages 9–11 (locked champion eval)
python src/training/stage9_explanation_consistency.py --n 200 --decode expected_cutpoints
python src/training/stage10_uncertainty.py --n-passes 30 --decode expected_cutpoints
python src/training/stage11_ablation_report.py
```
