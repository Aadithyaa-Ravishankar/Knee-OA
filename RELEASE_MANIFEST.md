# Release manifest — champion pipeline (Stage 6 + Stage 10 clinical)

This branch ships the **production path only**: frozen Stage 6 dual-stream cascade champion and the Stage 10 clinical deploy layer (temperature + severe-protect + abstain). Failed experiment scripts (Stages 7–9b, Stage 4b, deep cascade) stay local and are **not** included.

## Included

### Model
- `outputs/models/CHAMPION_stage6_cascade.pth` — best weights (test QWK **0.8417**)

### Core code
- `src/models/baseline.py`, `dual_stream.py`, `dual_stream_cascade.py`
- `src/losses/ordinal_loss.py`
- `src/evaluation/metrics.py`, `abstain.py`, `ensemble.py`
- `src/preprocessing/dataset.py`, `dataloader.py`, `transforms.py`, `compute_stats.py`, `build_combined_dataset.py`
- Training: `stage3` → `stage4` → `stage5` → `stage6` (reproduce champion)
- Deploy: `stage10_clinical_deploy.py`

### Reports
- `reports/champion_model.json`
- `reports/stage2b/normalization_stats_combined.json`
- `reports/stage6/*` (summary, history, figures)
- `reports/stage10/stage10_clinical_deploy.json`

## Not included (by design)
- `Dataset/` / `Dataset_combined/` (see DATASET_SETUP.md)
- Failed stages: 4b, 7, 8, 9a, 9b, deep-cascade model
- Large baseline `.pth` files (~270MB)
- One-shot Grad-CAM / Stage 1 audit extras beyond existing tracked outputs

## Quick use
```bash
# Clinical deploy (no training)
python src/training/stage10_clinical_deploy.py

# Or load champion
# outputs/models/CHAMPION_stage6_cascade.pth
```

## Metrics (test, N=1656)
| Card | QWK | Notes |
|------|-----|-------|
| Champion (raw S6) | 0.8417 | All cases |
| Clinical covered | 0.8777 | 81.8% coverage (301 abstain) |
