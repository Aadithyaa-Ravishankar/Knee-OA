# Pipeline Phase 0/1 — champion and candidates on validation (bootstrap 95% CI)

n = 826 validation images. Test split not loaded.

| Model / decode | QWK [95% CI] | Macro-F1 | Acc | KL0→KL1 | KL2→KL1 | Pred KL1 / true KL1 | ΔQWK vs champion [CI] | Sig? |
|---|---|---:|---:|---:|---:|---:|---|---|
| stage6_champion | 0.8011 [0.770, 0.830] | 0.6708 | 0.6138 | 102 | 75 | 263 / 153 | — | — |
| stage6_champion+flipTTA | 0.8167 [0.788, 0.844] | 0.6848 | 0.6392 | 100 | 67 | 259 / 153 | +0.0156 [+0.0012, +0.0302] | yes |
| champion+TTA+argmax | 0.8186 [0.790, 0.846] | 0.6784 | 0.6368 | 93 | 73 | 257 / 153 | +0.0175 [+0.0027, +0.0327] | yes |
| champion+TTA+prior_correct_argmax | 0.7901 [0.757, 0.822] | 0.6398 | 0.6719 | 9 | 7 | 26 / 153 | -0.0110 [-0.0330, +0.0098] | no |
| champion+TTA+expected_grade_cutpoints(crossfit) | 0.8134 [0.785, 0.839] | 0.6549 | 0.6114 | 109 | 83 | 287 / 153 | +0.0124 [-0.0036, +0.0269] | no |
| champion+TTA+prior_correct+cutpoints(crossfit) | 0.8148 [0.787, 0.842] | 0.6626 | 0.6174 | 108 | 68 | 262 / 153 | +0.0137 [-0.0023, +0.0300] | no |
| stage9b_sampler_ft | 0.8163 [0.787, 0.844] | 0.6789 | 0.6404 | 92 | 65 | 244 / 153 | +0.0152 [-0.0027, +0.0322] | no |
| stage9b_sampler_ft+flipTTA | 0.8280 [0.798, 0.857] | 0.6989 | 0.6610 | 88 | 68 | 245 / 153 | +0.0269 [+0.0059, +0.0464] | yes |
| stage16_t0x2 | 0.8068 [0.779, 0.833] | 0.6438 | 0.5823 | 136 | 119 | 373 / 153 | +0.0057 [-0.0118, +0.0233] | no |
| stage16_t0x2+flipTTA | 0.8081 [0.779, 0.835] | 0.6506 | 0.5908 | 131 | 118 | 365 / 153 | +0.0070 [-0.0098, +0.0237] | no |
| stage18_armA | 0.8071 [0.776, 0.838] | 0.6855 | 0.6368 | 106 | 72 | 281 / 153 | +0.0060 [-0.0119, +0.0240] | no |
| stage18_armA+flipTTA | 0.8201 [0.792, 0.847] | 0.7001 | 0.6392 | 114 | 75 | 292 / 153 | +0.0190 [+0.0013, +0.0371] | yes |

- Champion OA-threshold rank-inconsistency rate: **0.0169**
- Stage 6 sampler prior: [0.0957, 0.2872, 0.1596, 0.2128, 0.2447]  vs natural Chen prior: [0.3956, 0.181, 0.2624, 0.131, 0.0299]
- Cutpoint variants are cross-fit: cutpoints for each patient group are fit on the other groups.
