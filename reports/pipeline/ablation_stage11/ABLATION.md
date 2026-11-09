# Stage 11 — Ablation report (artifact-based)

Built from locked-test finals, Stage 6 joint-focus screen, and Stage 9/10 reports when present. **Does not retrain or overwrite the champion.**

| Config | Decode | Test QWK | Macro-F1 | Acc | OOF QWK | Global band | ESS | Flip rate | ECE (MC) |
|--------|--------|---------:|---------:|----:|--------:|------------:|----:|----------:|---------:|
| sqrt + OARSI aux (no attn prior) | `native` | 0.8557 | 0.7117 | 0.6920 | 0.8402 | 0.455 | — | — | — |
| aux + soft attn / spatial prior (champion weights) | `native` | 0.8613 | 0.7085 | 0.6878 | 0.8386 | 0.751 | 0.947 | 0.110 | — |
| champion + OOF expected-grade cutpoints (final decode) | `expected_cutpoints` | 0.8673 | 0.6953 | 0.6914 | 0.8386 | 0.751 | 0.947 | 0.110 | 0.0208 |

## Reading the table

1. **aux_only → attn_spatial:** joint-focus prior lifts global band mass (localization) while keeping competitive QWK.
2. **attn_spatial → cutpoints:** Stage 7 expected-grade cutpoints raise locked-test QWK to the reported **0.8673** champion.
3. **ESS / flip / ECE:** filled from Stages 9–10 on the champion when those reports exist.

## Sources

- `aux_only`: `reports\pipeline\final_test_r50_dual_cascade_sqrt_aux.json`
- `attn_spatial`: `reports\pipeline\final_test_r50_dual_cascade_sqrt_aux_attn_spatial.json`
- `attn_spatial_cutpoints`: `reports\pipeline\final_test_r50_dual_cascade_sqrt_aux_attn_spatial_expected_cutpoints.json`
