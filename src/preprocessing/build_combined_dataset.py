"""
build_combined_dataset.py
=========================
Option A+: Chen/OAI train + G&P MedicalExpert-I KL-2, KL-3, KL-4.
Val / test / auto_test stay Chen-only.

Also runs validation gates (counts, sizes, no G&P leakage into val/test).

Usage (from project root):
    python src/preprocessing/build_combined_dataset.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHEN_ROOT = PROJECT_ROOT / "Dataset"
GP_ROOT = Path(r"D:\NN & DL\Digital Knee X-ray Images\Knee X-ray Images")
GP_EXPERT_I = GP_ROOT / "MedicalExpert-I" / "MedicalExpert-I"
OUT_ROOT = PROJECT_ROOT / "Dataset_combined"
REPORT_DIR = PROJECT_ROOT / "reports" / "stage2b"
IMG_SIZE = 224

# Expected train counts after Option A+ (KL-2/3/4 from G&P)
EXPECTED_TRAIN = {0: 2286, 1: 1046, 2: 1748, 3: 978, 4: 379}
EXPECTED_TRAIN_TOTAL = 6437
EXPECTED_VAL = {0: 328, 1: 153, 2: 212, 3: 106, 4: 27}
EXPECTED_TEST = {0: 639, 1: 296, 2: 447, 3: 223, 4: 51}

GP_GRADE_DIRS = {
    2: "2Mild",
    3: "3Moderate",
    4: "4Severe",
}
EXPECTED_GP_ADDED = {2: 232, 3: 221, 4: 206}


def count_pngs(split_dir: Path) -> dict[int, int]:
    counts = {g: 0 for g in range(5)}
    for g in range(5):
        d = split_dir / str(g)
        if d.exists():
            counts[g] = sum(1 for p in d.iterdir() if p.suffix.lower() == ".png")
    return counts


def copy_chen_split(split: str) -> None:
    src = CHEN_ROOT / split
    dst = OUT_ROOT / split
    if not src.exists():
        raise FileNotFoundError(f"Missing Chen split: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def add_gp_minority() -> dict[int, int]:
    """Resize G&P Expert-I KL-2/3/4 to 224 and add into train/{2,3,4}."""
    added = {g: 0 for g in GP_GRADE_DIRS}
    for grade, folder in GP_GRADE_DIRS.items():
        src_dir = GP_EXPERT_I / folder
        dst_dir = OUT_ROOT / "train" / str(grade)
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing G&P folder: {src_dir}")

        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".png")
        for fpath in tqdm(files, desc=f"  G&P KL-{grade}", ncols=80):
            out_name = f"gp_E1_{fpath.stem}.png"
            out_path = dst_dir / out_name
            img = Image.open(fpath).convert("L")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            img.save(out_path)
            added[grade] += 1
    return added


def gate_check() -> tuple[bool, list[dict]]:
    """Return (all_passed, list of gate results)."""
    gates: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        gates.append({"gate": name, "pass": ok, "detail": detail})

    # Sources exist
    add("chen_exists", CHEN_ROOT.exists(), str(CHEN_ROOT))
    add("gp_expert_i_exists", GP_EXPERT_I.exists(), str(GP_EXPERT_I))
    add("combined_exists", OUT_ROOT.exists(), str(OUT_ROOT))

    train_c = count_pngs(OUT_ROOT / "train")
    val_c = count_pngs(OUT_ROOT / "val")
    test_c = count_pngs(OUT_ROOT / "test")

    add(
        "train_counts",
        train_c == EXPECTED_TRAIN,
        f"got={train_c} expected={EXPECTED_TRAIN}",
    )
    add(
        "train_total",
        sum(train_c.values()) == EXPECTED_TRAIN_TOTAL,
        f"got={sum(train_c.values())} expected={EXPECTED_TRAIN_TOTAL}",
    )
    add("val_unchanged", val_c == EXPECTED_VAL, f"got={val_c}")
    add("test_unchanged", test_c == EXPECTED_TEST, f"got={test_c}")

    # No gp_ files in val/test
    leak_val = list((OUT_ROOT / "val").rglob("gp_*.png"))
    leak_test = list((OUT_ROOT / "test").rglob("gp_*.png"))
    add("no_gp_in_val", len(leak_val) == 0, f"found={len(leak_val)}")
    add("no_gp_in_test", len(leak_test) == 0, f"found={len(leak_test)}")

    # G&P files present in train 2/3/4
    for g, n_exp in EXPECTED_GP_ADDED.items():
        n = len(list((OUT_ROOT / "train" / str(g)).glob("gp_*.png")))
        add(f"gp_kl{g}_added", n == n_exp, f"got={n} expected={n_exp}")

    # Spot-check dimensions (sample up to 20 gp + 20 chen)
    bad_size = []
    samples = []
    samples += list((OUT_ROOT / "train" / "3").glob("gp_*.png"))[:20]
    samples += list((OUT_ROOT / "train" / "0").glob("*.png"))[:20]
    for p in samples:
        with Image.open(p) as im:
            if im.size != (IMG_SIZE, IMG_SIZE):
                bad_size.append(f"{p.name}:{im.size}")
    add("image_size_224", len(bad_size) == 0, f"bad={bad_size[:5]}")

    # Imbalance improved vs original Chen train KL-4=173
    ratio = train_c[0] / max(train_c[4], 1)
    add(
        "imbalance_improved",
        ratio < 10.0,
        f"KL0/KL4 ratio={ratio:.2f} (was ~13.2)",
    )

    # Train loader smoke (optional soft gate)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.preprocessing.dataloader import build_dataloaders

        loaders = build_dataloaders(
            root_dir=OUT_ROOT,
            batch_size=8,
            num_workers=0,
            channels=3,
            use_weighted_sampler=True,
            splits=("train", "val"),
            pin_memory=False,
        )
        imgs, labels = next(iter(loaders["train"]))
        ok_shape = tuple(imgs.shape) == (8, 3, 224, 224)
        add(
            "dataloader_smoke",
            ok_shape and len(labels) == 8,
            f"batch_shape={tuple(imgs.shape)} labels={labels.tolist()}",
        )
    except Exception as e:
        add("dataloader_smoke", False, f"error={e}")

    return all(g["pass"] for g in gates), gates


def main() -> None:
    print("=" * 60)
    print("  BUILD Dataset_combined (Option A) + GATES")
    print("=" * 60)

    if not CHEN_ROOT.exists():
        print(f"[FAIL] Chen dataset missing: {CHEN_ROOT}")
        sys.exit(1)
    if not GP_EXPERT_I.exists():
        print(f"[FAIL] G&P Expert-I missing: {GP_EXPERT_I}")
        sys.exit(1)

    if OUT_ROOT.exists():
        print(f"\n[1/4] Removing old {OUT_ROOT} ...")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[2/4] Copying Chen splits (train/val/test/auto_test)...")
    for split in ("train", "val", "test", "auto_test"):
        print(f"  copying {split}...")
        copy_chen_split(split)

    print("\n[3/4] Adding G&P Expert-I KL-2/3/4 (resize → 224×224)...")
    added = add_gp_minority()
    print(f"  added: KL-2={added[2]}, KL-3={added[3]}, KL-4={added[4]}")

    print("\n[4/4] Running gates...")
    ok, gates = gate_check()
    for g in gates:
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  [{status}] {g['gate']}: {g['detail']}")

    train_c = count_pngs(OUT_ROOT / "train")
    summary = {
        "option": "Aplus_chen_train_plus_gp_kl2_kl3_kl4",
        "output": str(OUT_ROOT),
        "train_counts": train_c,
        "train_total": sum(train_c.values()),
        "gp_added": added,
        "gates": gates,
        "all_gates_passed": ok,
    }
    out_json = REPORT_DIR / "combined_dataset_gates.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_json}")

    print("\n" + "=" * 60)
    if ok:
        print("  ALL GATES PASSED")
        print(f"  Train total = {sum(train_c.values())}")
        print(f"  Use dataset root: {OUT_ROOT}")
        print("=" * 60)
        sys.exit(0)
    print("  SOME GATES FAILED — fix issues before training")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
