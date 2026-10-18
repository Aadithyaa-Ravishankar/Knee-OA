"""
stage1_dataset_audit.py
=======================
Stage 1: Comprehensive Dataset Audit for Knee OA KL-Grade Dataset.

This script performs a NON-DESTRUCTIVE audit of the dataset.
It does NOT modify, rename, delete, or overwrite any original image file.

Usage:
    python stage1_dataset_audit.py

Outputs are saved to:
    knee_oa_xai/outputs/figures/   — charts and sample X-rays
    knee_oa_xai/outputs/metrics/   — CSV files and audit report
"""

import os
import sys
import re
import hashlib
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = Path("/Users/aadithyaaravishankar/Desktop/KNEE OA/dataset")
FIGURES_DIR  = BASE_DIR / "outputs" / "figures"
METRICS_DIR  = BASE_DIR / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS      = ["train", "val", "test", "auto_test"]
KL_GRADES   = [0, 1, 2, 3, 4]
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def file_md5(path: Path) -> str:
    """Return MD5 hex-digest of a file for duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_filename(stem: str):
    """
    Parse a filename stem like '9001695L' or '9001695R'.
    Returns (patient_id, side) or (None, None) if no match.
    """
    m = re.match(r"^(\d+)([LR])$", stem.strip())
    if m:
        return m.group(1), m.group(2)
    return None, None


def collect_records() -> pd.DataFrame:
    """Walk every split/grade folder and return a tidy DataFrame of all images."""
    records = []
    for split in SPLITS:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            print(f"  [WARN] Split '{split}' not found — skipping.")
            continue
        for grade in KL_GRADES:
            grade_dir = split_dir / str(grade)
            if not grade_dir.exists():
                print(f"  [WARN] Grade dir '{split}/{grade}' not found — skipping.")
                continue
            for fpath in grade_dir.iterdir():
                if fpath.suffix.lower() not in IMG_EXTS:
                    continue
                patient_id, side = parse_filename(fpath.stem)
                records.append({
                    "path":       str(fpath),
                    "split":      split,
                    "kl_grade":   grade,
                    "filename":   fpath.name,
                    "stem":       fpath.stem,
                    "ext":        fpath.suffix.lower(),
                    "patient_id": patient_id,
                    "side":       side,
                    "size_bytes": fpath.stat().st_size,
                })
    return pd.DataFrame(records)


def probe_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Open each image and record width, height, mode, and corruption status.
    This is READ-ONLY — images are never modified.
    """
    widths, heights, modes, corrupted = [], [], [], []
    print("\nProbing image properties (read-only)...")
    for path_str in tqdm(df["path"], unit="img", ncols=80):
        try:
            with Image.open(path_str) as img:
                w, h = img.size
                m = img.mode
            widths.append(w)
            heights.append(h)
            modes.append(m)
            corrupted.append(False)
        except (UnidentifiedImageError, Exception):
            widths.append(None)
            heights.append(None)
            modes.append(None)
            corrupted.append(True)

    df = df.copy()
    df["width"]     = widths
    df["height"]    = heights
    df["img_mode"]  = modes
    df["corrupted"] = corrupted
    df["aspect_ratio"] = df.apply(
        lambda r: round(r["width"] / r["height"], 3)
                  if (r["width"] and r["height"]) else None,
        axis=1,
    )
    return df


def compute_md5(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MD5 hashes for duplicate detection."""
    hashes = []
    print("\nComputing MD5 hashes for duplicate detection...")
    for path_str in tqdm(df["path"], unit="img", ncols=80):
        try:
            hashes.append(file_md5(Path(path_str)))
        except Exception:
            hashes.append(None)
    df = df.copy()
    df["md5"] = hashes
    return df


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    """Bar charts of KL-grade distribution per split and overall."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("KL-Grade Class Distribution", fontsize=16, fontweight="bold")

    # ── Overall ──
    overall = df.groupby("kl_grade").size().reset_index(name="count")
    ax = axes[0]
    bars = ax.bar(overall["kl_grade"].astype(str),
                  overall["count"],
                  color=sns.color_palette("viridis", 5),
                  edgecolor="white", linewidth=0.8)
    ax.set_title("Overall Dataset", fontsize=13)
    ax.set_xlabel("KL Grade")
    ax.set_ylabel("Number of Images")
    for bar, val in zip(bars, overall["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15, str(val),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── Per Split ──
    ax = axes[1]
    pivot = (df.groupby(["split", "kl_grade"])
               .size()
               .unstack(fill_value=0))
    pivot.plot(kind="bar", ax=ax,
               color=sns.color_palette("viridis", 5),
               edgecolor="white", linewidth=0.5)
    ax.set_title("Per Split", fontsize=13)
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of Images")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.legend(title="KL Grade", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_sample_xrays(df: pd.DataFrame):
    """Show one representative X-ray from each KL grade (from train split)."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("Representative X-ray Samples per KL Grade (Train Split)",
                 fontsize=14, fontweight="bold")

    train_df = df[df["split"] == "train"]
    for ax, grade in zip(axes, KL_GRADES):
        grade_imgs = train_df[train_df["kl_grade"] == grade]
        if grade_imgs.empty:
            ax.set_title(f"KL-{grade}\n(no images)")
            ax.axis("off")
            continue
        sample = grade_imgs.sample(1, random_state=42).iloc[0]
        try:
            img = Image.open(sample["path"]).convert("L")
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"KL-{grade}\n{sample['filename']}\n"
                f"{img.width}×{img.height}",
                fontsize=9,
            )
        except Exception as e:
            ax.set_title(f"KL-{grade}\n[ERROR]")
        ax.axis("off")

    plt.tight_layout()
    out = FIGURES_DIR / "sample_xrays_per_grade.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_dimension_stats(df: pd.DataFrame):
    """Scatter plot of width vs. height, coloured by KL grade."""
    valid = df[~df["corrupted"]].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Image Dimension Statistics", fontsize=14, fontweight="bold")

    palette = sns.color_palette("viridis", 5)

    # Width distribution
    ax = axes[0]
    for grade in KL_GRADES:
        sub = valid[valid["kl_grade"] == grade]["width"].dropna()
        ax.hist(sub, bins=30, alpha=0.6, label=f"KL-{grade}", color=palette[grade])
    ax.set_title("Width Distribution")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

    # Height distribution
    ax = axes[1]
    for grade in KL_GRADES:
        sub = valid[valid["kl_grade"] == grade]["height"].dropna()
        ax.hist(sub, bins=30, alpha=0.6, label=f"KL-{grade}", color=palette[grade])
    ax.set_title("Height Distribution")
    ax.set_xlabel("Height (px)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

    # Aspect ratio distribution
    ax = axes[2]
    for grade in KL_GRADES:
        sub = valid[valid["kl_grade"] == grade]["aspect_ratio"].dropna()
        ax.hist(sub, bins=30, alpha=0.6, label=f"KL-{grade}", color=palette[grade])
    ax.set_title("Aspect Ratio Distribution")
    ax.set_xlabel("Width / Height")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "dimension_stats.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_heatmap_split_grade(df: pd.DataFrame):
    """Heatmap of image counts for split × KL grade."""
    pivot = df.groupby(["split", "kl_grade"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Image Count"})
    ax.set_title("Image Count: Split × KL Grade", fontsize=13, fontweight="bold")
    ax.set_xlabel("KL Grade")
    ax.set_ylabel("Split")
    plt.tight_layout()
    out = FIGURES_DIR / "split_grade_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# LEAKAGE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_leakage(df: pd.DataFrame) -> dict:
    """
    Check whether a patient appears in multiple splits.
    Returns a dict with findings.
    """
    has_ids = df["patient_id"].notna().sum()
    if has_ids == 0:
        return {"patient_ids_found": False}

    # Patient → splits mapping
    patient_splits = (
        df[df["patient_id"].notna()]
        .groupby("patient_id")["split"]
        .apply(lambda x: sorted(set(x)))
    )
    leaked = patient_splits[patient_splits.apply(len) > 1]

    return {
        "patient_ids_found":     True,
        "total_unique_patients": df["patient_id"].nunique(),
        "patients_with_L_knee":  (df["side"] == "L").sum(),
        "patients_with_R_knee":  (df["side"] == "R").sum(),
        "patients_with_both":    int(
            df.groupby("patient_id")["side"].apply(
                lambda s: set(s.dropna()) == {"L", "R"}
            ).sum()
        ),
        "leaked_patients_count": len(leaked),
        "leaked_patients":       leaked.reset_index().to_dict(orient="records")[:20],
    }


# ──────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, leakage: dict, duplicates_df: pd.DataFrame) -> str:
    """Compose the final audit report as a markdown string."""
    valid = df[~df["corrupted"]]
    total = len(df)
    n_corrupted = df["corrupted"].sum()

    grade_dist = df.groupby("kl_grade").size()
    split_grade = df.groupby(["split", "kl_grade"]).size().unstack(fill_value=0)

    dim_summary = valid.groupby("kl_grade")[["width", "height"]].agg(
        ["mean", "std", "min", "max"]
    ).round(1)

    modes = valid["img_mode"].value_counts().to_dict()
    exts  = df["ext"].value_counts().to_dict()

    n_dups     = duplicates_df["md5"].nunique() if not duplicates_df.empty else 0
    dup_images = len(duplicates_df) if not duplicates_df.empty else 0

    # Imbalance ratio
    min_cls  = grade_dist.min()
    max_cls  = grade_dist.max()
    imb_ratio = round(max_cls / min_cls, 2)

    lines = []
    lines.append("# Stage 1 — Dataset Audit Report")
    lines.append("**Project:** Anatomy-Aware Explainable AI for Knee OA Severity Classification")
    lines.append(f"**Dataset Path:** `{DATASET_DIR}`\n")

    lines.append("---")
    lines.append("## 1. Dataset Size")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total images found | **{total:,}** |")
    lines.append(f"| Corrupted images   | {n_corrupted} |")
    lines.append(f"| Valid images       | {len(valid):,} |")
    lines.append(f"| File extensions    | {exts} |")
    lines.append(f"| Image modes (valid)| {modes} |")

    lines.append("\n## 2. KL-Grade Distribution (Overall)")
    lines.append("| KL Grade | Count | % of Total |")
    lines.append("|----------|-------|------------|")
    for g in KL_GRADES:
        cnt = grade_dist.get(g, 0)
        pct = round(cnt / total * 100, 1)
        lines.append(f"| KL-{g} | {cnt:,} | {pct}% |")
    lines.append(f"| **TOTAL** | **{total:,}** | 100% |")

    lines.append("\n## 3. Dataset Structure (Split × Grade)")
    lines.append(split_grade.to_markdown())

    lines.append("\n## 4. Image Characteristics")
    lines.append("### Dimension Statistics (mean ± std) per KL Grade")
    lines.append(dim_summary.to_markdown())

    unique_dims = valid[["width", "height"]].drop_duplicates()
    lines.append(f"\n**Unique (width × height) combinations found:** {len(unique_dims)}")

    ar_mean = valid["aspect_ratio"].mean()
    ar_std  = valid["aspect_ratio"].std()
    lines.append(f"**Aspect ratio (mean ± std):** {ar_mean:.3f} ± {ar_std:.3f}")

    lines.append("\n## 5. Class Imbalance")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Most frequent class  | KL-{grade_dist.idxmax()} ({grade_dist.max():,} images) |")
    lines.append(f"| Least frequent class | KL-{grade_dist.idxmin()} ({grade_dist.min():,} images) |")
    lines.append(f"| Imbalance ratio (max/min) | **{imb_ratio}×** |")
    lines.append("\n**Recommended strategies (Stage 2):**")
    lines.append("- Weighted random sampling during DataLoader construction")
    lines.append("- Focal Loss or Class-Weighted Cross-Entropy")
    lines.append("- Oversampling minority classes with augmentation")
    lines.append("- Ordinal-aware loss that accounts for class proximity\n")

    lines.append("## 6. Data-Quality Issues")
    lines.append(f"- Corrupted / unreadable images: **{n_corrupted}**")
    lines.append(f"- Zero-byte files: **{(df['size_bytes'] == 0).sum()}**")

    lines.append("\n## 7. Duplicate Findings")
    lines.append(f"- Unique duplicate MD5 hashes found: **{n_dups}**")
    lines.append(f"- Total duplicate images: **{dup_images}**")
    if not duplicates_df.empty:
        lines.append(f"\n> ⚠️ Duplicates detected! Review `{METRICS_DIR / 'duplicates.csv'}`")
    else:
        lines.append("> ✅ No exact duplicate images found.")

    lines.append("\n## 8. Patient / Subject ID Findings")
    if leakage.get("patient_ids_found"):
        lines.append(f"- **Filename pattern:** `<PatientID><L|R>.ext` — e.g. `9001695L.png`")
        lines.append(f"- Unique patient IDs: **{leakage['total_unique_patients']:,}**")
        lines.append(f"- Left-knee images (L): **{leakage['patients_with_L_knee']:,}**")
        lines.append(f"- Right-knee images (R): **{leakage['patients_with_R_knee']:,}**")
        lines.append(f"- Patients with both knees in dataset: **{leakage['patients_with_both']:,}**")
    else:
        lines.append("- Patient IDs **not parseable** from filenames.")

    lines.append("\n## 9. Potential Data Leakage")
    if leakage.get("patient_ids_found"):
        n_leaked = leakage["leaked_patients_count"]
        if n_leaked > 0:
            lines.append(f"> ⚠️ **{n_leaked} patients appear in MORE THAN ONE split** — DATA LEAKAGE RISK!")
            lines.append("> Review `metrics/leaked_patients.csv` for details.")
        else:
            lines.append("> ✅ No patients found in multiple splits — no immediate leakage detected.")
        lines.append("\n**Important note:** Both L and R knee images from the same patient "
                     "may appear in the same split. This is acceptable, but ensure they "
                     "are NOT split across train/test at the image level.")
    else:
        lines.append("> ⚠️ Patient IDs are unavailable — image-level random split may cause leakage. "
                     "Recommend explicit patient-level split if IDs become available.")

    lines.append("\n## 10. Recommended Train / Validation / Test Strategy")
    lines.append("The dataset **already provides pre-defined splits** (train / val / test / auto_test).")
    lines.append("However:")
    lines.append("- A patient-level split is **strongly recommended** to prevent leakage.")
    lines.append("- Group all images from the same patient into the same split.")
    lines.append("- If the existing Kaggle splits already do this, they can be used directly.")
    lines.append("- Verify: no patient ID appears in both `train` and `val` or `test`.\n")

    lines.append("## 11. Recommended Next Steps (Stage 2)")
    lines.append("1. Confirm whether existing splits are patient-level (manual verification)")
    lines.append("2. Implement image preprocessing pipeline (resize, normalize, CLAHE)")
    lines.append("3. Implement PyTorch `Dataset` and `DataLoader` with class-weighted sampling")
    lines.append("4. Establish baseline: single-stream ResNet-50 classifier with standard CE loss")
    lines.append("5. Evaluate baseline with Quadratic Weighted Kappa and macro-F1")
    lines.append("6. Only then proceed to dual-stream + ordinal loss + Grad-CAM\n")

    lines.append("---")
    lines.append("*Report auto-generated by `stage1_dataset_audit.py`*")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  STAGE 1 — KNEE OA DATASET AUDIT")
    print("=" * 65)

    # 1. Collect all image records
    print("\n[1/7] Scanning dataset directory...")
    df = collect_records()
    print(f"  Found {len(df):,} image files across {df['split'].nunique()} splits.")

    # 2. Probe image properties (read-only)
    print("\n[2/7] Probing image properties...")
    df = probe_images(df)

    # 3. Compute MD5 hashes for duplicate detection
    print("\n[3/7] Computing MD5 hashes...")
    df = compute_md5(df)

    # Identify duplicates
    hash_counts = df.groupby("md5")["path"].count()
    dup_hashes  = hash_counts[hash_counts > 1].index.tolist()
    duplicates_df = df[df["md5"].isin(dup_hashes)].sort_values("md5")

    # 4. Leakage analysis
    print("\n[4/7] Analysing patient-level data leakage...")
    leakage = analyze_leakage(df)

    # 5. Save CSVs
    print("\n[5/7] Saving CSV records...")
    df.to_csv(METRICS_DIR / "full_inventory.csv", index=False)
    print(f"  Saved → {METRICS_DIR / 'full_inventory.csv'}")

    if not duplicates_df.empty:
        duplicates_df.to_csv(METRICS_DIR / "duplicates.csv", index=False)
        print(f"  Saved → {METRICS_DIR / 'duplicates.csv'}")

    # Leakage CSV
    if leakage.get("leaked_patients_count", 0) > 0:
        pd.DataFrame(leakage["leaked_patients"]).to_csv(
            METRICS_DIR / "leaked_patients.csv", index=False
        )
        print(f"  Saved → {METRICS_DIR / 'leaked_patients.csv'}")

    # 6. Visualizations
    print("\n[6/7] Generating visualizations...")
    plot_class_distribution(df)
    plot_sample_xrays(df)
    plot_dimension_stats(df)
    plot_heatmap_split_grade(df)

    # 7. Build and save report
    print("\n[7/7] Writing audit report...")
    report_md = build_report(df, leakage, duplicates_df)
    report_path = METRICS_DIR / "audit_report.md"
    report_path.write_text(report_md)
    print(f"  Saved → {report_path}")

    # Print a condensed console summary
    print("\n" + "=" * 65)
    print("  AUDIT SUMMARY")
    print("=" * 65)
    grade_dist = df.groupby("kl_grade").size()
    print(f"  Total images   : {len(df):,}")
    print(f"  Corrupted      : {df['corrupted'].sum()}")
    print(f"  Duplicates     : {len(duplicates_df)}")
    print()
    print("  KL-Grade Distribution:")
    for g in KL_GRADES:
        cnt = grade_dist.get(g, 0)
        bar = "█" * (cnt // 100)
        print(f"    KL-{g}: {cnt:>5,}  {bar}")
    print()
    if leakage.get("patient_ids_found"):
        print(f"  Unique patients   : {leakage['total_unique_patients']:,}")
        print(f"  Leaked patients   : {leakage['leaked_patients_count']}")
    print()
    print(f"  Figures  → {FIGURES_DIR}")
    print(f"  Metrics  → {METRICS_DIR}")
    print("=" * 65)
    print("\n✅ Stage 1 complete. Review the audit report before proceeding to Stage 2.")


if __name__ == "__main__":
    main()
