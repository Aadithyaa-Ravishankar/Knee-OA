# Grad-CAM proof — correctly graded images

- run: `r50_dual_cascade_sqrt_aux_attn_spatial`  **split: test**
- decode: `expected_cutpoints`
- n correct panels: **40** (stratified across KL when capped)
- pred: 5-fold ensemble; CAM: [0]
- mean combined / global / local band: **0.779 / 0.747 / 0.910** (uniform baseline 0.278)

High band mass on **correct** grades supports that the model attends to the joint region when it grades right — not only on error cases.

## By true KL

| KL | n | band_combined | band_local | mean P(true) |
|---:|---:|---:|---:|---:|
| 0 | 8 | 0.791 | 0.946 | 0.765 |
| 1 | 8 | 0.788 | 0.902 | 0.512 |
| 2 | 8 | 0.673 | 0.881 | 0.626 |
| 3 | 8 | 0.783 | 0.883 | 0.953 |
| 4 | 8 | 0.858 | 0.937 | 0.802 |

## How to read

- Panels: `panels/KL{k}/*_panel.png` = original | combined | global | local.
- Target = predicted (= true) class.
- Band mass >> uniform baseline → attention concentrated on the joint band.

