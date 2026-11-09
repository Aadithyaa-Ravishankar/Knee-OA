# Decode calibration — `r50_dual_cascade_sqrt_aux_attn_spatial` (OOF only)

n=6604, folds=5. Test untouched.

| decode | QWK | Macro-F1 | Acc | KL0→1 | KL2→1 | Recall KL0..4 |
|---|---:|---:|---:|---:|---:|---|
| crossfit_expected_cutpoints | 0.8434 | 0.6741 | 0.6655 | 505 | 431 | 0.78/0.45/0.58/0.75/0.94 |
| native_oa0.50 | 0.8386 | 0.6919 | 0.6687 | 502 | 490 | 0.78/0.47/0.56/0.78/0.85 |
| crossfit_oa_threshold | 0.8386 | 0.6919 | 0.6687 | 502 | 490 | 0.78/0.47/0.56/0.78/0.85 |

- Cross-fit OA thresholds by fold: f0=0.50, f1=0.50, f2=0.50, f3=0.50, f4=0.50
- Mean cross-fit OA threshold ≈ **0.50** (in-sample best thr=0.50, optimistic QWK=0.8386)

Fair scores are the **crossfit_*** rows. Use those for selection; apply the mean OA threshold (or re-fit cutpoints on all OOF) only at locked test if OOF improves.
