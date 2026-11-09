# Correct vs misgraded — final model

Model: `r50_dual_cascade_sqrt_aux_attn_spatial` (5-fold).

- **OOF fold-val** = images scored by the fold that did **not** train on them (training-time / selection set; not the locked test).
- **Locked test** = held-out test split (ensemble of all 5 folds + flip TTA).
- **FINAL decode** = expected-grade cutpoints.

## Summary

| Split | Decode | n | Correct | Misgraded | Acc | QWK | Neighbor wrong | Far wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OOF fold-val | native | 6604 | 4416 | 2188 | 66.9% | 0.8386 | 1944 | 244 |
| OOF fold-val | cutpoints | 6604 | 4454 | 2150 | 67.4% | 0.8450 | 1900 | 250 |
| Locked test | native | 1656 | 1139 | 517 | 68.8% | 0.8613 | 475 | 42 |
| Locked test | cutpoints | 1656 | 1145 | 511 | 69.1% | 0.8673 | 469 | 42 |

## Per-KL (FINAL = locked test cutpoints)

| KL | n | Correct | Wrong | Recall |
|---:|---:|---:|---:|---:|
| 0 | 639 | 520 | 119 | 81.4% |
| 1 | 296 | 118 | 178 | 39.9% |
| 2 | 447 | 279 | 168 | 62.4% |
| 3 | 223 | 179 | 44 | 80.3% |
| 4 | 51 | 49 | 2 | 96.1% |

## Per-KL (OOF cutpoints)

| KL | n | Correct | Wrong | Recall |
|---:|---:|---:|---:|---:|
| 0 | 2614 | 2055 | 559 | 78.6% |
| 1 | 1199 | 494 | 705 | 41.2% |
| 2 | 1728 | 1056 | 672 | 61.1% |
| 3 | 863 | 660 | 203 | 76.5% |
| 4 | 200 | 189 | 11 | 94.5% |
