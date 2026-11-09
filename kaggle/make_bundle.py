"""
make_bundle.py
==============
Build the two zip files to upload to Kaggle as Datasets:

  kaggle/bundle/knee-oa-data.zip   Dataset_combined/{train,val}/<grade>/*.png  (Chen only:
                                   no gp_* files, no test/, no auto_test/)
  kaggle/bundle/knee-oa-code.zip   knee_oa_code/src/**.py + folds.csv + folds_features.csv

The test split is deliberately NOT uploaded: the final locked test is run locally
with stage7_final_test.py after downloading the fold models.

Usage:
  python kaggle/make_bundle.py            # both zips (Dataset_combined)
  python kaggle/make_bundle.py --code     # only the code zip (after code changes)
  python kaggle/make_bundle.py --data-root path/to/kneeKL299   # remap folds by filename
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Dataset_combined"
FOLDS_CSV = PROJECT_ROOT / "reports" / "pipeline" / "folds.csv"
FEATURES_CSV = PROJECT_ROOT / "reports" / "pipeline" / "folds_features.csv"
MEDLAT_MAP = PROJECT_ROOT / "reports" / "pipeline" / "medlat_side_mapping.json"
OUT = PROJECT_ROOT / "kaggle" / "bundle"


def path_map_by_filename(data_root: Path, fold_paths: list[str]) -> dict[str, str]:
    """Map folds.csv relative paths -> relative paths under data_root (same basename)."""
    index: dict[str, str] = {}
    for split in ("train", "val"):
        for grade in range(5):
            d = Path(data_root) / split / str(grade)
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"} and not f.name.lower().startswith("gp_"):
                    index[f.name] = f"{split}/{grade}/{f.name}"
    mapped, missing = {}, []
    for p in fold_paths:
        hit = index.get(Path(p).name)
        if hit is None:
            missing.append(p)
        else:
            mapped[p] = hit
    print(f"data-root map: {len(mapped)}/{len(fold_paths)} fold rows matched under {data_root}")
    if missing:
        raise SystemExit(f"{len(missing)} fold images missing under {data_root}, e.g. {missing[:3]}")
    return mapped


def build_data(data_root: Path) -> Path:
    with FOLDS_CSV.open(encoding="utf-8") as fh:
        fold_paths = [r["path"] for r in csv.DictReader(fh)]
    bad = [p for p in fold_paths if Path(p).name.lower().startswith("gp_") or not p.startswith(("train/", "val/"))]
    if bad:
        raise SystemExit(f"folds.csv has {len(bad)} non-Chen or non-train/val rows, e.g. {bad[:3]}")
    mapped = path_map_by_filename(data_root, fold_paths)
    out = OUT / "knee-oa-data.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # PNGs are already compressed
        for src_rel, dst_rel in mapped.items():
            # Keep Dataset_combined/ prefix in the zip so Kaggle notebooks stay unchanged;
            # file bytes come from data_root (e.g. kneeKL299).
            zf.write(data_root / dst_rel, f"Dataset_combined/{dst_rel}")
    print(f"data: {len(mapped)} images from {data_root} -> {out} ({out.stat().st_size / 2**20:.0f} MB)")
    return out


def build_code() -> Path:
    out = OUT / "knee-oa-code.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted((PROJECT_ROOT / "src").rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            zf.write(f, f"knee_oa_code/{f.relative_to(PROJECT_ROOT).as_posix()}")
            n += 1
        zf.write(FOLDS_CSV, "knee_oa_code/reports/pipeline/folds.csv")
        if not FEATURES_CSV.is_file():
            raise SystemExit(f"missing {FEATURES_CSV}; run src/preprocessing/make_fold_features.py first")
        zf.write(FEATURES_CSV, "knee_oa_code/reports/pipeline/folds_features.csv")
        if MEDLAT_MAP.is_file():
            zf.write(MEDLAT_MAP, "knee_oa_code/reports/pipeline/medlat_side_mapping.json")
    print(f"code: {n} .py files + folds.csv + folds_features.csv -> {out} ({out.stat().st_size / 2**10:.0f} KB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true", help="only rebuild the code zip")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT,
                    help="image root with train/val/<grade>/... (default Dataset_combined; "
                         "use kneeKL299 for native 299px)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.code:
        build_data(args.data_root)
    build_code()


if __name__ == "__main__":
    main()
