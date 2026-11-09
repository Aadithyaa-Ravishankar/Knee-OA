# Stage 9 — Explanation Consistency

- run: `r50_dual_cascade_sqrt_aux_attn_spatial`  split: `test`  decode: `expected_cutpoints`
- n: **200**  CAM fold: `0`  pred folds: `[0, 1, 2, 3, 4]`
- perturbations: brightness ±5%, Gaussian noise σ=0.01

## Population metrics

- **mean AFS (combined / global / local):** 0.777 / 0.727 / 0.926 (uniform baseline 0.278)
- **frac AFS > 0.5:** 0.970  (target mean AFS > 0.5 → PASS)
- **mean ESS (SSIM):** 0.947  (frac > 0.85: 0.980; target mean > 0.85 → PASS)
- **grade flip rate:** 0.110  (target < 0.05 → FAIL)

AFS uses the same joint-band energy as Stage 8 `band_mass`. ESS is mean SSIM between original and perturbed combined CAMs. Grade flip counts any hard-grade change under the same perturbations.
