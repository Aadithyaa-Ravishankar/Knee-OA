# Pipeline OOF report (fold-validation only; test untouched)

| Run [decode] | folds | n | QWK [95% CI] | Macro-F1 | Acc | Recall KL0..KL4 | KL0→1 | KL2→1 | ΔQWK vs baseline [CI] |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| r50_dual_cascade_sqrt_aux_attn_spatial_gfocus [native] | 5 | 6604 | 0.8405 [0.830, 0.850] | 0.6882 | 0.6693 | 0.80/0.45/0.54/0.77/0.88 | 446 | 508 | — |
| r50_dual_cascade_sqrt_aux_attn_spatial_gfocus [cutpoints] | 5 | 6604 | 0.8399 [0.830, 0.849] | 0.6718 | 0.6576 | 0.78/0.43/0.54/0.76/0.95 | 500 | 452 | — |

`*` = 95% CI of the difference excludes 0. The Δ column is computed only on images both runs predicted (same folds), so it is the fair comparison when runs have different numbers of folds done; the QWK column is not. QWK CIs and paired Δ use **patient-clustered** bootstrap when `patients` is present in oof.npz (resampling patients, keeping all of their images).
