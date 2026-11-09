# Aux JSN/OST agreement — correct vs wrong KL (OOF)

Run: `r50_dual_cascade_sqrt_aux_attn_spatial` (fold OOF). True aux from `folds_features.csv`.
Aux grade = count of cumulative thresholds with prob > 0.5.
KL wrong uses **final cutpoint decode** (same cuts as locked test).

## Why this matters

If aux JSN/OST match labels **equally well** on KL-wrong images, the trunk still sees joint features —
KL errors are mostly ordinal/boundary, not “ignored JSN/ost”.
If aux collapses on KL-wrong images, the model is missing feature signal on those cases.

## Cutpoint decode (final)

### all (n=6604)

- mean feature exact-acc: **0.717**  mean MAE: **0.316**
- max(JSN med,lat) exact: **0.757** (n=6604)
- max(OST med,lat) exact: **0.543** (n=4143)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 6604 | 0.785 | 0.219 | 0.810 |
| aux_jsn_lat | 6604 | 0.949 | 0.055 | 0.873 |
| aux_ost_med | 4143 | 0.533 | 0.527 | 0.632 |
| aux_ost_lat | 4143 | 0.599 | 0.462 | 0.645 |

### KL correct (n=4454)

- mean feature exact-acc: **0.746**  mean MAE: **0.289**
- max(JSN med,lat) exact: **0.842** (n=4454)
- max(OST med,lat) exact: **0.593** (n=2646)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 4454 | 0.857 | 0.145 | 0.888 |
| aux_jsn_lat | 4454 | 0.963 | 0.039 | 0.921 |
| aux_ost_med | 2646 | 0.560 | 0.504 | 0.671 |
| aux_ost_lat | 2646 | 0.605 | 0.466 | 0.666 |

### KL wrong (n=2150)

- mean feature exact-acc: **0.658**  mean MAE: **0.371**
- max(JSN med,lat) exact: **0.581** (n=2150)
- max(OST med,lat) exact: **0.454** (n=1497)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 2150 | 0.637 | 0.371 | 0.566 |
| aux_jsn_lat | 2150 | 0.920 | 0.087 | 0.732 |
| aux_ost_med | 1497 | 0.486 | 0.569 | 0.490 |
| aux_ost_lat | 1497 | 0.589 | 0.456 | 0.558 |

### KL neighbor wrong (n=1900)

- mean feature exact-acc: **0.669**  mean MAE: **0.358**
- max(JSN med,lat) exact: **0.587** (n=1900)
- max(OST med,lat) exact: **0.493** (n=1293)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 1900 | 0.636 | 0.369 | 0.591 |
| aux_jsn_lat | 1900 | 0.929 | 0.074 | 0.793 |
| aux_ost_med | 1293 | 0.503 | 0.551 | 0.531 |
| aux_ost_lat | 1293 | 0.606 | 0.436 | 0.607 |

### KL far wrong (n=250)

- mean feature exact-acc: **0.588**  mean MAE: **0.459**
- max(JSN med,lat) exact: **0.536** (n=250)
- max(OST med,lat) exact: **0.206** (n=204)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 250 | 0.644 | 0.380 | 0.184 |
| aux_jsn_lat | 250 | 0.856 | 0.188 | 0.254 |
| aux_ost_med | 204 | 0.373 | 0.686 | -0.193 |
| aux_ost_lat | 204 | 0.480 | 0.583 | -0.091 |

## Summary: correct vs wrong (cutpoints)

| subset | n | mean aux exact-acc | mean MAE | max-JSN acc | max-OST acc |
|---|---:|---:|---:|---:|---:|
| all | 6604 | 0.717 | 0.316 | 0.757 | 0.543 |
| KL correct | 4454 | 0.746 | 0.289 | 0.842 | 0.593 |
| KL wrong | 2150 | 0.658 | 0.371 | 0.581 | 0.454 |
| KL neighbor wrong | 1900 | 0.669 | 0.358 | 0.587 | 0.493 |
| KL far wrong | 250 | 0.588 | 0.459 | 0.536 | 0.206 |

## Native decode (reference)

### all (n=6604)

- mean feature exact-acc: **0.717**  mean MAE: **0.316**
- max(JSN med,lat) exact: **0.757** (n=6604)
- max(OST med,lat) exact: **0.543** (n=4143)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 6604 | 0.785 | 0.219 | 0.810 |
| aux_jsn_lat | 6604 | 0.949 | 0.055 | 0.873 |
| aux_ost_med | 4143 | 0.533 | 0.527 | 0.632 |
| aux_ost_lat | 4143 | 0.599 | 0.462 | 0.645 |

### KL correct (n=4416)

- mean feature exact-acc: **0.745**  mean MAE: **0.291**
- max(JSN med,lat) exact: **0.844** (n=4416)
- max(OST med,lat) exact: **0.586** (n=2613)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 4416 | 0.858 | 0.144 | 0.890 |
| aux_jsn_lat | 4416 | 0.964 | 0.038 | 0.924 |
| aux_ost_med | 2613 | 0.554 | 0.513 | 0.676 |
| aux_ost_lat | 2613 | 0.606 | 0.467 | 0.674 |

### KL wrong (n=2188)

- mean feature exact-acc: **0.661**  mean MAE: **0.366**
- max(JSN med,lat) exact: **0.581** (n=2188)
- max(OST med,lat) exact: **0.469** (n=1530)

| feature | n | exact-acc | MAE | QWK |
|---|---:|---:|---:|---:|
| aux_jsn_med | 2188 | 0.638 | 0.369 | 0.563 |
| aux_jsn_lat | 2188 | 0.920 | 0.089 | 0.729 |
| aux_ost_med | 1530 | 0.498 | 0.552 | 0.443 |
| aux_ost_lat | 1530 | 0.587 | 0.454 | 0.505 |

