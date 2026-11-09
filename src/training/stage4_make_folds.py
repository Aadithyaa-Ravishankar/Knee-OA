"""
stage4_make_folds.py
=====================
Build patient-grouped, label-stratified folds over Chen train+val (G&P excluded).
The test split is not included and stays locked for one final evaluation.

Usage:
    python src/training/stage4_make_folds.py [--data-root Dataset_combined] [--n-folds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.protocol import collect_chen_rows, make_folds, write_folds_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "pipeline" / "folds.csv")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = make_folds(collect_chen_rows(args.data_root), args.n_folds, args.seed)
    write_folds_csv(rows, args.out)

    by_fold = {f: Counter(r["label"] for r in rows if r["fold"] == f) for f in range(args.n_folds)}
    patients_per_fold = {f: {r["patient"] for r in rows if r["fold"] == f} for f in range(args.n_folds)}
    overlap = sum(
        len(patients_per_fold[a] & patients_per_fold[b])
        for a in range(args.n_folds) for b in range(a + 1, args.n_folds)
    )
    summary = {
        "n_images": len(rows),
        "n_patients": len({r["patient"] for r in rows}),
        "gp_images_included": sum(r["path"].split("/")[-1].startswith("gp_") for r in rows),
        "patients_shared_between_folds": overlap,
        "per_fold_class_counts": {f: [by_fold[f][k] for k in range(5)] for f in by_fold},
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"folds → {args.out}")


if __name__ == "__main__":
    main()
