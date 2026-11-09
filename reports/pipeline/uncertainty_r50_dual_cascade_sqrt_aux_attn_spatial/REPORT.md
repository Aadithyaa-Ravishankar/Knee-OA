# Stage 10 — Uncertainty & Calibration

- run: `r50_dual_cascade_sqrt_aux_attn_spatial`  folds: `[0, 1, 2, 3, 4]`  MC passes: **30**
- decode: `expected_cutpoints`  n_test: **1656**
- non-finite AMP draws excluded from entropy/confidence means: **24**

## MC Dropout (primary)

- **ECE:** 0.0208  (15 bins)
- **mean predictive entropy:** 0.6768
- **mean confidence (P[pred]):** 0.6933
- **accuracy:** 0.6902
- entropy by true KL: {'0': 0.6074112578060519, '1': 0.8723836086086325, '2': 0.7878799592655243, '3': 0.43820314467654903, '4': 0.3065173685662793}

## Deterministic ensemble (reference)

- **ECE:** 0.019956275429306693
- **accuracy:** 0.6890096618357487

## Artifacts

- `reliability_mc.png` / `reliability_deterministic.png`
- `summary.json`, `rows.jsonl`

Dropout layers are forced to train mode; BatchNorm stays in eval. Fold probabilities are averaged, then hard grades use Stage 7 cutpoints.
