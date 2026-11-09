# Error analysis — final model (attn_spatial + expected-grade cutpoints)

Locked test only. Weights unchanged; decode = OOF-fit expected-grade cutpoints.

## Bottom line

Looking at the joint is **mostly solved** on mistakes. Remaining errors are almost all
**neighbor KL grades** (0↔1, 1↔2, 2↔3) — soft clinical boundaries / hard visual cues,
not “watching the wrong place.”

## Locked-test scorecard

| | native decode | **cutpoints (final)** |
|---|---:|---:|
| QWK | 0.8613 | **0.8673** |
| Accuracy | 0.6878 | 0.6914 |
| Errors / n | 517 / 1656 | **511 / 1656** |

- Neighbor errors (|Δ|=1): **469** (91.8% of errors)
- Far errors (|Δ|≥2): **42** (8.2% of errors)

## Confusion pairs (true → pred), cutpoints

| pair | n | share of errors |
|---|---:|---:|
| 1→0 | 116 | 22.7% |
| 0→1 | 107 | 20.9% |
| 2→1 | 103 | 20.2% |
| 1→2 | 62 | 12.1% |
| 2→3 | 37 | 7.2% |
| 2→0 | 28 | 5.5% |
| 3→2 | 21 | 4.1% |
| 3→4 | 21 | 4.1% |
| 0→2 | 12 | 2.3% |
| 3→1 | 2 | 0.4% |
| 4→3 | 2 | 0.4% |

Largest buckets: **1→0**, **0→1**, **2→1**, **1→2** (early OA boundary).

## Are we still looking at the wrong region?

Prior Grad-CAM on **all native locked-test misgrades** for the same weights
(`gradcam_errors_attn_spatial_test/`):

- mean band combined / global / local: **0.787 / 0.746 / 0.926**
- uniform chance baseline ≈ 0.278
- conclusion in that report: joint focus is strong on errors; leftovers are ordinal boundary calls

Cutpoint decode only changes how soft scores become KL labels — **it does not move Grad-CAM**.
So those localization numbers still apply to “why wrong after focusing on the joint.”

## Why correct region ≠ correct grade

1. **KL is ordinal and fuzzy.** KL0↔1 = doubtful osteophytes; KL1↔2 = mild JSN. Radiologists disagree on these pairs; the model inherits that ambiguity.
2. **Low decision margin.** On native error CAMs, mean P(true) was often ~0.23–0.30 while P(pred) ~0.5–0.7 — borderline, not confident wrong anatomy.
3. **KL1 is the hard class.** Cutpoints recall KL1 ≈ 0.40; many true-1 go to 0 or 2.
4. **Projection X-ray limits.** Subtle osteophytes / compartment JSN can be weak or view-dependent even when attention is on the joint line.
5. **Far misses are rare** (~8%) — catastrophic wrong-place / wrong-severity failures are not the main story.

## What would still help (grading, not CAM)

- Soft KL1↔2 training (`--soft-kl12`), or more OARSI/aux signal on borderline cases
- Accept irreducible 0↔1 / 1↔2 noise; report QWK + pair rates, not only accuracy
- Do **not** expect more global-stream CAM work to fix these grades

## Confusion matrix (cutpoints)

Rows = true, cols = pred:

```
[[520, 107, 12, 0, 0], [116, 118, 62, 0, 0], [28, 103, 279, 37, 0], [0, 2, 21, 179, 21], [0, 0, 0, 2, 49]]
```

