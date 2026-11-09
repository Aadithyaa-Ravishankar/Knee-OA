# Knee-OA pipeline stages (canonical numbering)

Use this numbering in docs, theses, and talk tracks. Old “Stage 19” experiment labels are retired.

| Stage | Name | Code / artifacts |
|------:|------|------------------|
| **1** | Dataset audit | `src/preprocessing/stage1_dataset_audit.py` |
| **2** | Preprocessing | transforms, loaders, `Dataset_combined`, norms |
| **3** | Early dual-cascade / baseline era | `src/models/dual_stream*.py`, `stage3_phase01_champion.py` |
| **4** | Patient-grouped 5-fold CV | `stage4_make_folds.py`, `stage4_cv_train.py`, `stage4_run_folds.py`, `stage4_oof_report.py` |
| **5** | OARSI aux multi-task | `stage5_dump_aux.py`, aux CSV / spatial aux flags |
| **6** | Joint-focus champion weights | `r50_dual_cascade_sqrt_aux_attn_spatial` under `outputs/cv/runs/` |
| **7** | Decode calibration + locked test | `stage7_decode_calibrate.py`, `stage7_final_test.py` (expected-grade cutpoints) |
| **8** | Explainability (Grad-CAM) | `stage8_gradcam.py`, `reports/pipeline/gradcam_*` |
| **9** | Explanation consistency | `stage9_explanation_consistency.py` — AFS/`band_mass`, ESS (SSIM), grade flip |
| **10** | Uncertainty & calibration | `stage10_uncertainty.py` — MC Dropout (N=30), entropy, ECE, reliability diagrams |
| **11** | Ablation reporting | `stage11_ablation_report.py` — artifact table (no champion overwrite) |

> Stages **9–11** evaluate the locked Stage 6–8 champion. They do **not** retrain or replace cutpoints.

## Final reported model

- **Weights:** Stage 6 — `outputs/cv/runs/r50_dual_cascade_sqrt_aux_attn_spatial/`
- **Decode:** Stage 7 — OOF-fit expected-grade cutpoints
- **Locked-test QWK:** **0.8673**
- **Report:** `reports/pipeline/final_test_r50_dual_cascade_sqrt_aux_attn_spatial_expected_cutpoints.json`
- **XAI proof:** `reports/pipeline/gradcam_correct_attn_spatial_cutpoints/`
- **Consistency:** `reports/pipeline/explanation_consistency_r50_dual_cascade_sqrt_aux_attn_spatial/`
- **Uncertainty:** `reports/pipeline/uncertainty_r50_dual_cascade_sqrt_aux_attn_spatial/`
- **Ablation:** `reports/pipeline/ablation_stage11/`

## Path map (after cleanup)

| Role | Path |
|------|------|
| Reports / folds / finals | `reports/pipeline/` |
| CV fold weights | `outputs/cv/runs/` |
| Kaggle working runs dir | `cv_runs/` (zip: `results_knee_oa.zip`) |
