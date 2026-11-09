# Aux feature eval — `r50_dual_cascade_sqrt_aux`

Source: `outputs\cv\runs\r50_dual_cascade_sqrt_aux` (`aux_oof.npz`). N dump images = 2644; matched to features = 2644.

## Per-feature grades (cumulative P>0.5 count vs folds_features.csv)

Non-missing feature rows only.

### `aux_jsn_med`

- n=2644  accuracy=0.8222  weighted κ=0.8395

| true\pred | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 1621 | 93 | 0 | 0 |
| 1 | 279 | 289 | 25 | 0 |
| 2 | 5 | 56 | 217 | 5 |
| 3 | 0 | 0 | 7 | 47 |

### `aux_jsn_lat`

- n=2644  accuracy=0.9527  weighted κ=0.8561

| true\pred | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 2411 | 9 | 1 | 0 |
| 1 | 80 | 40 | 7 | 0 |
| 2 | 6 | 16 | 45 | 2 |
| 3 | 1 | 1 | 2 | 23 |

### `aux_ost_med`

- n=1660  accuracy=0.6313  weighted κ=0.7086

| true\pred | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 453 | 110 | 1 | 0 |
| 1 | 196 | 459 | 24 | 3 |
| 2 | 7 | 121 | 47 | 19 |
| 3 | 6 | 52 | 73 | 89 |

### `aux_ost_lat`

- n=1660  accuracy=0.6717  weighted κ=0.6838

| true\pred | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 776 | 71 | 2 | 0 |
| 1 | 236 | 255 | 10 | 3 |
| 2 | 17 | 100 | 30 | 5 |
| 3 | 7 | 52 | 42 | 54 |

## KL-from-features (predicted aux grades + KL probs)

Fold-wise: train LogisticRegression on other folds' OOF, predict held-out fold.
Features = hard aux grades (4) concatenated with native KL softmax/cascade probs (5).

| decoder | n | QWK | accuracy | recall KL0 | KL1 | KL2 | KL3 | KL4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KL-from-features | 2644 | 0.8318 | 0.7012 | 0.8843 | 0.1938 | 0.7197 | 0.7775 | 0.8625 |
| native KL head | 2644 | 0.8437 | 0.6872 | 0.8499 | 0.4292 | 0.5564 | 0.7746 | 0.8625 |

### Confusion — KL-from-features

| true\pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 925 | 71 | 46 | 4 | 0 |
| 1 | 266 | 93 | 117 | 4 | 0 |
| 2 | 97 | 60 | 498 | 37 | 0 |
| 3 | 3 | 3 | 64 | 269 | 7 |
| 4 | 0 | 0 | 0 | 11 | 69 |

### Confusion — native KL head (same images)

| true\pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 889 | 139 | 15 | 3 | 0 |
| 1 | 221 | 206 | 50 | 3 | 0 |
| 2 | 71 | 200 | 385 | 36 | 0 |
| 3 | 1 | 12 | 55 | 268 | 10 |
| 4 | 0 | 0 | 0 | 11 | 69 |

