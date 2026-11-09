# Medial / lateral side check

Measure = dark joint-gap height in each half of the central 0.72 band. Narrowed side = **smaller gap**. The chosen mapping comes from **medial-only** JSN (direct medial signal). Lateral-only is a cross-check.

## Medial-only (jsn_med≥1, jsn_lat=0, KL 3–4) — chooses the rule

- Candidates: 793; measured: 782 (R=368, L=414)
- R knees with smaller gap on image-left: 89/368 (24.2%)
- L knees with smaller gap on image-left: 333/414 (80.4%)
- R majority agreement: 75.8%
- Medial-only implies `medial_is_image_left_for_r` = False

## Lateral-only (jsn_lat≥1, jsn_med=0, KL 3–4) — cross-check

- Candidates: 227; measured: 210 (R=125, L=85)
- R knees with smaller gap on image-left: 44/125 (35.2%)
- L knees with smaller gap on image-left: 56/85 (65.9%)
- Lateral-only implies `medial_is_image_left_for_r` = True
- Cohorts agree: False

## Chosen rule

- **For R knees, medial = image-right half**
- `medial_is_image_left_for_r` = False (from medial-only)
- L consistent with R mirror on medial-only: True
- Ambiguous (stop before medlat train if true): False

Mapping JSON: `reports\pipeline\medlat_side_mapping.json`
Contact sheet (medial-only tiles first): `reports\pipeline\medlat_side_contact_sheet.png`
