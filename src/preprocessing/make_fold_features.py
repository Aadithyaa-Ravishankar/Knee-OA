"""
make_fold_features.py
=====================
Attach four OARSI targets to the Pipeline train/val folds.

Reads reports/pipeline/folds.csv and the Dataset_combined rows of
data/oai_labels/oai_features_matched.csv. Test and auto_test are not written.

Usage:
  python src/preprocessing/make_fold_features.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLDS = PROJECT_ROOT / "reports" / "pipeline" / "folds.csv"
MATCHED = PROJECT_ROOT / "data" / "oai_labels" / "oai_features_matched.csv"
OUT = PROJECT_ROOT / "reports" / "pipeline" / "folds_features.csv"
AUX = ("aux_jsn_med", "aux_jsn_lat", "aux_ost_med", "aux_ost_lat")
FIELDS = ("path", "label", "patient", "source_split", "fold", *AUX)


def _num(cell: str) -> int | None:
    text = (cell or "").strip()
    if text == "" or text.lower() == "nan":
        return None
    value = float(text)
    if value not in (0, 1, 2, 3):
        raise SystemExit(f"feature grade {cell!r} is outside 0-3")
    return int(value)


def _max_pair(a: str, b: str) -> str:
    vals = [v for v in (_num(a), _num(b)) if v is not None]
    return "" if not vals else str(max(vals))


def _rel(path: str) -> str:
    text = path.replace("\\", "/")
    marker = "Dataset_combined/"
    at = text.find(marker)
    if at < 0:
        raise SystemExit(f"not a Dataset_combined path: {path}")
    return text[at + len(marker):]


def _self_check() -> None:
    assert _max_pair("", "") == ""
    assert _max_pair("1", "") == "1" and _max_pair("", "0") == "0"
    assert _max_pair("1", "3") == "3"
    assert _rel(r"D:\NN & DL\Knee-OA\Dataset_combined\train\0\9001695L.png") == "train/0/9001695L.png"


def main() -> None:
    _self_check()
    with FOLDS.open(encoding="utf-8", newline="") as fh:
        folds = list(csv.DictReader(fh))
    fold_of = {r["path"]: r["fold"] for r in folds}

    features = {}
    with MATCHED.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] != "Dataset_combined":
                continue
            key = _rel(row["path"])
            if key in features:
                raise SystemExit(f"duplicate Dataset_combined path {key}")
            features[key] = row

    out_rows = []
    for src in folds:
        feat = features.get(src["path"])
        if feat is None:
            raise SystemExit(f"no OAI row for {src['path']}")
        out_rows.append({
            "path": src["path"],
            "label": src["label"],
            "patient": src["patient"],
            "source_split": src["source_split"],
            "fold": src["fold"],
            "aux_jsn_med": "" if _num(feat["jsn_med"]) is None else str(_num(feat["jsn_med"])),
            "aux_jsn_lat": "" if _num(feat["jsn_lat"]) is None else str(_num(feat["jsn_lat"])),
            "aux_ost_med": _max_pair(feat["ost_fem_med"], feat["ost_tib_med"]),
            "aux_ost_lat": _max_pair(feat["ost_fem_lat"], feat["ost_tib_lat"]),
            "KL_folder": feat["KL_folder"],
        })

    if len(out_rows) != 6604:
        raise SystemExit(f"expected 6604 rows, got {len(out_rows)}")
    for row in out_rows:
        if not row["path"].startswith(("train/", "val/")):
            raise SystemExit(f"path is not train/val: {row['path']}")
        if Path(row["path"]).name.lower().startswith("gp_"):
            raise SystemExit(f"gp_ image in folds: {row['path']}")
        if int(row["label"]) != int(row["KL_folder"]):
            raise SystemExit(f"label {row['label']} != KL_folder {row['KL_folder']} for {row['path']}")
        if row["fold"] != fold_of[row["path"]]:
            raise SystemExit(f"fold changed for {row['path']}")
        for col in AUX:
            if row[col] not in ("", "0", "1", "2", "3"):
                raise SystemExit(f"{col}={row[col]!r} on {row['path']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDS} for r in out_rows)

    print(f"wrote {len(out_rows)} rows -> {OUT}")
    for col in AUX:
        print(f"\n{col} value counts: {dict(Counter(r[col] if r[col] else 'blank' for r in out_rows))}")
        for grade in range(5):
            group = [r for r in out_rows if int(r["label"]) == grade]
            missing = sum(r[col] == "" for r in group)
            rate = missing / len(group) if group else 0.0
            print(f"  KL{grade}: missing {missing}/{len(group)} ({rate:.1%})")


if __name__ == "__main__":
    main()
