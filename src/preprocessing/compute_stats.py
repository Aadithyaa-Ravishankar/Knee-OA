"""
compute_stats.py
================
Stage 2 — Compute actual per-channel mean and standard deviation
from the TRAINING split only.

These values are used to initialize the normalization transform in
transforms.py. Run this ONCE and paste the printed values into
transforms.py (TRAIN_MEAN_* / TRAIN_STD_*).

Usage:
    python src/preprocessing/compute_stats.py

Output is also saved to:
    reports/stage2/normalization_stats.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # Knee-OA/
DATASET_DIR  = PROJECT_ROOT / "Dataset_combined"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "stage2b"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS   = {".png", ".jpg", ".jpeg"}
KL_GRADES  = [0, 1, 2, 3, 4]


def collect_train_paths() -> list[Path]:
    paths = []
    train_dir = DATASET_DIR / "train"
    for grade in KL_GRADES:
        grade_dir = train_dir / str(grade)
        if not grade_dir.exists():
            continue
        for fpath in sorted(grade_dir.iterdir()):
            if fpath.suffix.lower() in IMG_EXTS:
                paths.append(fpath)
    return paths


def compute_mean_std(paths: list[Path]) -> tuple[float, float]:
    """
    Welford's online algorithm for numerically stable mean/std.
    Works on grayscale images in [0, 1] range.
    """
    n    = 0
    mean = 0.0
    M2   = 0.0

    print(f"Computing mean/std over {len(paths):,} training images...")
    for fpath in tqdm(paths, unit="img", ncols=80):
        try:
            arr = np.array(Image.open(fpath).convert("L"), dtype=np.float32) / 255.0
            for pixel in arr.ravel():
                n    += 1
                delta = pixel - mean
                mean  += delta / n
                M2    += delta * (pixel - mean)
        except Exception as e:
            print(f"  [WARN] Skipping {fpath.name}: {e}")

    std = float(np.sqrt(M2 / (n - 1))) if n > 1 else 0.0
    return float(mean), std


def main():
    print("=" * 60)
    print("  STAGE 2 — COMPUTE NORMALIZATION STATISTICS")
    print("=" * 60)

    paths = collect_train_paths()
    if not paths:
        print("[ERROR] No training images found. Check DATASET_DIR.")
        sys.exit(1)

    mean, std = compute_mean_std(paths)

    print(f"\n  Training split statistics (grayscale, [0,1]):")
    print(f"  Mean : {mean:.6f}")
    print(f"  Std  : {std:.6f}")

    print(f"\n  Copy these values into transforms.py:")
    print(f"  TRAIN_MEAN_1CH = [{mean:.6f}]")
    print(f"  TRAIN_STD_1CH  = [{std:.6f}]")
    print(f"  TRAIN_MEAN_3CH = [{mean:.6f}, {mean:.6f}, {mean:.6f}]")
    print(f"  TRAIN_STD_3CH  = [{std:.6f}, {std:.6f}, {std:.6f}]")

    # Save to JSON
    stats = {
        "split":           "train",
        "total_pixels":    len(paths) * 224 * 224,
        "total_images":    len(paths),
        "mean_grayscale":  round(mean, 6),
        "std_grayscale":   round(std, 6),
        "mean_1ch":        [round(mean, 6)],
        "std_1ch":         [round(std, 6)],
        "mean_3ch":        [round(mean, 6)] * 3,
        "std_3ch":         [round(std, 6)] * 3,
    }
    out = REPORTS_DIR / "normalization_stats_combined.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out}")
    print("=" * 60)
    print("✅ Combined-train normalization stats computed.")


if __name__ == "__main__":
    main()
