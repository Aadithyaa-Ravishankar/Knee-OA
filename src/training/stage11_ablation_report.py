"""
stage11_ablation_report.py
==========================
Stage 11 — Component ablation table from existing artifacts (no retrain,
no champion overwrite).

Aggregates locked-test finals, joint-focus screen, and Stage 7/9/10 reports
when present into one markdown/json table.

Example:
  python src/training/stage11_ablation_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "reports" / "pipeline"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(blob: dict | None, *keys, default=None):
    if blob is None:
        return default
    cur = blob
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_rows() -> list[dict]:
    """One row per available configuration / decode variant."""
    rows: list[dict] = []

    # --- locked-test finals ---
    finals = [
        (
            "aux_only",
            "sqrt + OARSI aux (no attn prior)",
            "native",
            REPORTS / "final_test_r50_dual_cascade_sqrt_aux.json",
        ),
        (
            "attn_spatial",
            "aux + soft attn / spatial prior (champion weights)",
            "native",
            REPORTS / "final_test_r50_dual_cascade_sqrt_aux_attn_spatial.json",
        ),
        (
            "attn_spatial_cutpoints",
            "champion + OOF expected-grade cutpoints (final decode)",
            "expected_cutpoints",
            REPORTS / "final_test_r50_dual_cascade_sqrt_aux_attn_spatial_expected_cutpoints.json",
        ),
    ]
    for key, desc, decode, path in finals:
        blob = _load_json(path)
        if blob is None:
            continue
        rows.append(
            {
                "id": key,
                "description": desc,
                "run": blob.get("run"),
                "decode": decode,
                "source": str(path.relative_to(PROJECT_ROOT)),
                "qwk": _metric(blob, "metrics", "qwk"),
                "macro_f1": _metric(blob, "metrics", "macro_f1"),
                "accuracy": _metric(blob, "metrics", "accuracy"),
                "n_test": blob.get("n_test"),
                "mean_global_band_mass": None,
                "mean_ess": None,
                "grade_flip_rate": None,
                "ece_mc": None,
                "mean_entropy": None,
            }
        )

    # --- joint-focus screen (OOF QWK + band mass) ---
    screen = _load_json(REPORTS / "screen_attn_spatial.json")
    if screen:
        by_run = {r["run"]: r for r in screen.get("rows", [])}
        for row in rows:
            run = row.get("run")
            if run in by_run:
                row["oof_mean_qwk"] = by_run[run].get("mean_qwk")
                row["mean_global_band_mass"] = by_run[run].get("mean_global_band_mass")
                row["mean_local_band_mass"] = by_run[run].get("mean_local_band_mass")
                row["mean_global_border_mass"] = by_run[run].get("mean_global_border_mass")

    # --- Stage 9 consistency (champion) ---
    s9 = _load_json(REPORTS / "explanation_consistency_r50_dual_cascade_sqrt_aux_attn_spatial" / "summary.json")
    if s9:
        summary = s9.get("summary", {})
        for row in rows:
            if row["id"] in ("attn_spatial", "attn_spatial_cutpoints"):
                row["mean_ess"] = summary.get("mean_ess")
                row["grade_flip_rate"] = summary.get("grade_flip_rate")
                row["mean_afs_combined"] = summary.get("mean_afs_combined")
                if row["mean_global_band_mass"] is None:
                    row["mean_global_band_mass"] = summary.get("mean_afs_global")

    # --- Stage 10 uncertainty (champion) ---
    s10 = _load_json(REPORTS / "uncertainty_r50_dual_cascade_sqrt_aux_attn_spatial" / "summary.json")
    if s10:
        mc = s10.get("mc", {})
        target_id = (
            "attn_spatial_cutpoints"
            if s10.get("decode") == "expected_cutpoints"
            else "attn_spatial"
        )
        for row in rows:
            if row["id"] == target_id:
                row["ece_mc"] = mc.get("ece")
                row["mean_entropy"] = mc.get("mean_entropy")

    return rows


def write_report(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": 11,
        "note": (
            "Ablation from existing artifacts only — no retrain. "
            "Champion remains attn_spatial + expected_cutpoints."
        ),
        "rows": rows,
    }
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Stage 11 — Ablation report (artifact-based)",
        "",
        "Built from locked-test finals, Stage 6 joint-focus screen, and Stage 9/10 "
        "reports when present. **Does not retrain or overwrite the champion.**",
        "",
        "| Config | Decode | Test QWK | Macro-F1 | Acc | OOF QWK | Global band | ESS | Flip rate | ECE (MC) |",
        "|--------|--------|---------:|---------:|----:|--------:|------------:|----:|----------:|---------:|",
    ]
    for r in rows:
        def fmt(v, nd=4):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.{nd}f}"
            return str(v)

        lines.append(
            f"| {r['description']} | `{r['decode']}` | {fmt(r.get('qwk'))} | "
            f"{fmt(r.get('macro_f1'))} | {fmt(r.get('accuracy'))} | "
            f"{fmt(r.get('oof_mean_qwk'))} | {fmt(r.get('mean_global_band_mass'), 3)} | "
            f"{fmt(r.get('mean_ess'), 3)} | {fmt(r.get('grade_flip_rate'), 3)} | "
            f"{fmt(r.get('ece_mc'))} |"
        )

    lines += [
        "",
        "## Reading the table",
        "",
        "1. **aux_only → attn_spatial:** joint-focus prior lifts global band mass "
        "(localization) while keeping competitive QWK.",
        "2. **attn_spatial → cutpoints:** Stage 7 expected-grade cutpoints raise locked-test QWK "
        "to the reported **0.8673** champion.",
        "3. **ESS / flip / ECE:** filled from Stages 9–10 on the champion when those reports exist.",
        "",
        "## Sources",
        "",
    ]
    for r in rows:
        lines.append(f"- `{r['id']}`: `{r['source']}`")
    lines.append("")
    (out_dir / "ABLATION.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPORTS / "ablation_stage11",
    )
    args = ap.parse_args()
    rows = build_rows()
    if not rows:
        raise SystemExit("no ablation sources found under reports/pipeline/")
    write_report(rows, args.out)
    print(f"wrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
