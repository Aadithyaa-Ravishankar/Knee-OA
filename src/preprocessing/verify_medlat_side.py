"""
verify_medlat_side.py
=====================
Decide whether a right-knee medial compartment is the image-left or image-right
half, using one-sided JSN knees (KL3/KL4) from folds_features.csv.

Primary measure: height of the dark femur–tibia gap in each half of the central
joint band. Narrowed side = smaller gap.

Two cohorts:
  lateral-only : jsn_lat>=1, jsn_med==0  → smaller gap should be the lateral half
  medial-only  : jsn_med>=1, jsn_lat==0  → smaller gap should be the medial half

The chosen R/L mapping is taken from the medial-only cohort (direct medial signal).
Lateral-only is reported as a cross-check.

Usage:
  python src/preprocessing/verify_medlat_side.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = PROJECT_ROOT / "reports" / "pipeline" / "folds_features.csv"
DATA = PROJECT_ROOT / "Dataset_combined"
OUT_DIR = PROJECT_ROOT / "reports" / "pipeline"
FRAC = 0.72
MAPPING_JSON = OUT_DIR / "medlat_side_mapping.json"
REPORT_MD = OUT_DIR / "medlat_side_check.md"
SHEET = OUT_DIR / "medlat_side_contact_sheet.png"


def dark_gap_height(half: np.ndarray) -> float:
    """Rows in the darkest valley of the vertical intensity profile (joint gap)."""
    profile = half.mean(axis=1).astype(np.float64)
    if profile.size < 8:
        return float("nan")
    k = np.ones(5) / 5.0
    profile = np.convolve(profile, k, mode="same")
    lo, hi = int(0.25 * len(profile)), int(0.75 * len(profile))
    mid = profile[lo:hi]
    valley = lo + int(np.argmin(mid))
    floor = profile[valley]
    peak = max(profile[lo:valley].max(initial=floor), profile[valley:hi].max(initial=floor))
    thr = floor + 0.35 * max(peak - floor, 1.0)
    top = valley
    while top > lo and profile[top] <= thr:
        top -= 1
    bot = valley
    while bot < hi - 1 and profile[bot] <= thr:
        bot += 1
    return float(bot - top)


def band_halves(gray: np.ndarray, frac: float = FRAC) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    ch, cw = int(round(h * frac)), int(round(w * frac))
    top, left = (h - ch) // 2, (w - cw) // 2
    band = gray[top : top + ch, left : left + cw]
    mid = cw // 2
    return band[:, :mid], band[:, mid:]


def load_onesided(which: str) -> list[dict]:
    """which: 'lateral' (lat>=1, med==0) or 'medial' (med>=1, lat==0)."""
    rows = []
    with FEATURES.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["label"]) not in (3, 4):
                continue
            jsm, jsl = r["aux_jsn_med"].strip(), r["aux_jsn_lat"].strip()
            if jsm == "" or jsl == "":
                continue
            med, lat = int(float(jsm)), int(float(jsl))
            if which == "lateral" and not (med == 0 and lat >= 1):
                continue
            if which == "medial" and not (lat == 0 and med >= 1):
                continue
            side = Path(r["path"]).stem[-1].upper()
            if side not in ("L", "R"):
                continue
            rows.append({"path": r["path"], "label": int(r["label"]), "side": side,
                         "jsn_med": med, "jsn_lat": lat, "cohort": which})
    return rows


def measure(knees: list[dict]) -> dict:
    votes_r_left = votes_r = votes_l_left = votes_l = 0
    measured = []
    for rec in knees:
        img_path = DATA / rec["path"]
        if not img_path.is_file():
            continue
        gray = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
        left, right = band_halves(gray)
        gl, gr = dark_gap_height(left), dark_gap_height(right)
        if not np.isfinite(gl) or not np.isfinite(gr) or gl == gr:
            continue
        left_smaller = gl < gr
        measured.append({**rec, "gap_left": gl, "gap_right": gr, "left_smaller_gap": left_smaller})
        if rec["side"] == "R":
            votes_r += 1
            votes_r_left += int(left_smaller)
        else:
            votes_l += 1
            votes_l_left += int(left_smaller)
    return {
        "candidates": len(knees),
        "measured": measured,
        "n_R": votes_r,
        "n_L": votes_l,
        "R_left_smaller": votes_r_left,
        "L_left_smaller": votes_l_left,
        "R_frac_left_smaller": (votes_r_left / votes_r) if votes_r else None,
        "L_frac_left_smaller": (votes_l_left / votes_l) if votes_l else None,
        "R_majority": (max(votes_r_left, votes_r - votes_r_left) / votes_r) if votes_r else None,
    }


def write_sheet(tiles_src: list[dict], path: Path) -> None:
    tiles = []
    for side in ("R", "L"):
        for m in [x for x in tiles_src if x["side"] == side][:8]:
            gray = Image.open(DATA / m["path"]).convert("L")
            g = np.asarray(gray, dtype=np.uint8)
            rgb = ImageOps.colorize(gray, black="black", white="white").convert("RGB")
            draw = ImageDraw.Draw(rgb)
            h, w = g.shape
            ch, cw = int(round(h * FRAC)), int(round(w * FRAC))
            top, left0 = (h - ch) // 2, (w - cw) // 2
            mid = left0 + cw // 2
            draw.rectangle([left0, top, left0 + cw, top + ch], outline=(80, 80, 255), width=1)
            draw.line([(mid, top), (mid, top + ch)], fill=(80, 80, 255), width=1)
            if m["left_smaller_gap"]:
                draw.rectangle([left0, top, mid, top + 4], fill=(0, 220, 0))
            else:
                draw.rectangle([mid, top, left0 + cw, top + 4], fill=(0, 220, 0))
            tag = "med" if m["cohort"] == "medial" else "lat"
            draw.text(
                (2, 2),
                f"{Path(m['path']).name} {tag} KL{m['label']} L{m['gap_left']:.0f}/R{m['gap_right']:.0f}",
                fill=(255, 220, 0),
            )
            tiles.append(rgb.resize((224, 224)))
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 224, rows * 224), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 224, (i // cols) * 224))
    sheet.save(path)


def main() -> None:
    lat = measure(load_onesided("lateral"))
    med = measure(load_onesided("medial"))
    if med["n_R"] < 5:
        raise SystemExit(f"too few measured R medial-only knees ({med['n_R']})")

    # Medial-only: smaller gap = medial half.
    # If R knees usually have the smaller gap on image-left → medial is image-left.
    med_r_left_is_medial = med["R_frac_left_smaller"] > 0.5
    medial_is_image_left_for_r = bool(med_r_left_is_medial)

    # Lateral-only cross-check: smaller gap = lateral → implies the opposite for medial.
    lat_implies_medial_left = None
    if lat["n_R"]:
        lat_r_left_is_lateral = lat["R_frac_left_smaller"] > 0.5
        lat_implies_medial_left = not lat_r_left_is_lateral

    # L mirror check under the chosen rule: L medial = opposite of R medial.
    # If medial_is_image_left_for_r, L medial is image-right → L left_smaller should be rare on medial-only.
    l_consistent = None
    if med["n_L"]:
        rate = med["L_frac_left_smaller"]
        l_consistent = (rate <= 0.45) if medial_is_image_left_for_r else (rate >= 0.55)

    cohorts_agree = (
        lat_implies_medial_left is None or lat_implies_medial_left == medial_is_image_left_for_r
    )
    ambiguous = (med["R_majority"] or 0) < 0.6 or l_consistent is False

    mapping = {
        "medial_is_image_left_for_r": medial_is_image_left_for_r,
        "chosen_from": "medial_only",
        "rule": (
            "For R knees, medial = image-left half"
            if medial_is_image_left_for_r
            else "For R knees, medial = image-right half"
        ),
        "ambiguous": ambiguous,
        "cohorts_agree": cohorts_agree,
        "medial_only": {
            "n_candidates": med["candidates"],
            "n_measured": len(med["measured"]),
            "n_R": med["n_R"],
            "n_L": med["n_L"],
            "R_fraction_image_left_smaller_gap": med["R_frac_left_smaller"],
            "L_fraction_image_left_smaller_gap": med["L_frac_left_smaller"],
            "R_majority_agreement": med["R_majority"],
            "implies_medial_is_image_left_for_r": med_r_left_is_medial,
        },
        "lateral_only": {
            "n_candidates": lat["candidates"],
            "n_measured": len(lat["measured"]),
            "n_R": lat["n_R"],
            "n_L": lat["n_L"],
            "R_fraction_image_left_smaller_gap": lat["R_frac_left_smaller"],
            "L_fraction_image_left_smaller_gap": lat["L_frac_left_smaller"],
            "R_majority_agreement": lat["R_majority"],
            "implies_medial_is_image_left_for_r": lat_implies_medial_left,
        },
        "L_consistent_with_R_mirror_on_medial_only": l_consistent,
    }

    # Contact sheet: medial-only first (reason for the rule), then lateral-only
    write_sheet(med["measured"] + lat["measured"], SHEET)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_JSON.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    mo, lo = mapping["medial_only"], mapping["lateral_only"]
    lines = [
        "# Medial / lateral side check",
        "",
        "Measure = dark joint-gap height in each half of the central 0.72 band. "
        "Narrowed side = **smaller gap**. "
        "The chosen mapping comes from **medial-only** JSN (direct medial signal). "
        "Lateral-only is a cross-check.",
        "",
        "## Medial-only (jsn_med≥1, jsn_lat=0, KL 3–4) — chooses the rule",
        "",
        f"- Candidates: {mo['n_candidates']}; measured: {mo['n_measured']} "
        f"(R={mo['n_R']}, L={mo['n_L']})",
        f"- R knees with smaller gap on image-left: "
        f"{med['R_left_smaller']}/{mo['n_R']} ({mo['R_fraction_image_left_smaller_gap']:.1%})",
        (
            f"- L knees with smaller gap on image-left: "
            f"{med['L_left_smaller']}/{mo['n_L']} ({mo['L_fraction_image_left_smaller_gap']:.1%})"
            if mo["n_L"]
            else "- L knees: none measured"
        ),
        f"- R majority agreement: {mo['R_majority_agreement']:.1%}",
        f"- Medial-only implies `medial_is_image_left_for_r` = {mo['implies_medial_is_image_left_for_r']}",
        "",
        "## Lateral-only (jsn_lat≥1, jsn_med=0, KL 3–4) — cross-check",
        "",
        f"- Candidates: {lo['n_candidates']}; measured: {lo['n_measured']} "
        f"(R={lo['n_R']}, L={lo['n_L']})",
        (
            f"- R knees with smaller gap on image-left: "
            f"{lat['R_left_smaller']}/{lo['n_R']} ({lo['R_fraction_image_left_smaller_gap']:.1%})"
            if lo["n_R"]
            else "- R knees: none measured"
        ),
        (
            f"- L knees with smaller gap on image-left: "
            f"{lat['L_left_smaller']}/{lo['n_L']} ({lo['L_fraction_image_left_smaller_gap']:.1%})"
            if lo["n_L"]
            else "- L knees: none measured"
        ),
        f"- Lateral-only implies `medial_is_image_left_for_r` = {lo['implies_medial_is_image_left_for_r']}",
        f"- Cohorts agree: {cohorts_agree}",
        "",
        "## Chosen rule",
        "",
        f"- **{mapping['rule']}**",
        f"- `medial_is_image_left_for_r` = {medial_is_image_left_for_r} (from medial-only)",
        f"- L consistent with R mirror on medial-only: {l_consistent}",
        f"- Ambiguous (stop before medlat train if true): {ambiguous}",
        "",
        f"Mapping JSON: `{MAPPING_JSON.relative_to(PROJECT_ROOT)}`",
        f"Contact sheet (medial-only tiles first): `{SHEET.relative_to(PROJECT_ROOT)}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if ambiguous:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
