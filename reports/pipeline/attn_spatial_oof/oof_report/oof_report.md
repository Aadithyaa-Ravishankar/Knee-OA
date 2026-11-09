# Pipeline OOF report (fold-validation only; test untouched)

| Run [decode] | folds | n | QWK [95% CI] | Macro-F1 | Acc | Recall KL0..KL4 | KL0→1 | KL2→1 | ΔQWK vs baseline [CI] |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| r50_dual_cascade_sqrt_aux_attn_spatial [cutpoints] | 5 | 6604 | 0.8434 [0.834, 0.852] | 0.6741 | 0.6655 | 0.78/0.45/0.58/0.75/0.94 | 505 | 431 | — |
| r50_dual_cascade_sqrt_aux_attn_spatial [native] | 5 | 6604 | 0.8386 [0.829, 0.848] | 0.6919 | 0.6687 | 0.78/0.47/0.56/0.78/0.85 | 502 | 490 | — |

`*` = 95% CI of the difference excludes 0. The Δ column is computed only on images both runs predicted (same folds), so it is the fair comparison when runs have different numbers of folds done; the QWK column is not. QWK CIs and paired Δ use **patient-clustered** bootstrap when `patients` is present in oof.npz (resampling patients, keeping all of their images).
