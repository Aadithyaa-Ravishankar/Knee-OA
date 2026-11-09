"""
stage7_final_test.py
=====================
ONE-TIME locked test evaluation of a chosen Pipeline run (fold ensemble + flip TTA).
Run this only after the configuration was picked from the OOF report.

Usage:
  python src/training/stage7_final_test.py --run-name <name> --i-understand-this-is-the-final-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.decoding import apply_cutpoints, expected_grade, fit_cutpoints
from src.evaluation.protocol import bootstrap_ci, metric_bundle, patient_id
from src.models.ordinal_net import OrdinalNet
from src.preprocessing.dataset import IMG_EXTS, KL_GRADES, KneeOADataset
from src.training.stage4_cv_train import build_transforms, predict, side_code
from src.training.stage4_oof_report import load_run

ENSEMBLE_MODES = ("logits", "probs")
DECODE_MODES = ("native", "expected_cutpoints")
_CFG_KEYS = ("img_size", "stream_mode", "n_aux", "spatial_aux")


def _fit_oof_cutpoints(run_dir: Path) -> np.ndarray:
    """Fit expected-grade cutpoints on all OOF fold-val probs (never on test)."""
    r = load_run(run_dir)
    if r is None:
        raise SystemExit(f"no oof.npz under {run_dir}; need OOF to fit cutpoints")
    return fit_cutpoints(expected_grade(r["probs"]), r["labels"])


class TestSideDataset(Dataset):
    """Test images with optional L/R side for medlat / spatial-aux flip-TTA. Excludes gp_* files."""

    def __init__(self, root, transform, with_side: bool):
        self.base = KneeOADataset(root, "test", transform=None)
        self.samples = [
            (fpath, label)
            for fpath, label in self.base.samples
            if not fpath.name.lower().startswith("gp_")
        ]
        if not self.samples:
            raise SystemExit(f"no non-gp_ test images under {root}/test")
        self.transform = transform
        self.with_side = with_side

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fpath, label = self.samples[i]
        img = self.transform(Image.open(fpath).convert("L"))
        if self.with_side:
            return img, label, side_code(fpath.name)
        return img, label


def _fold_index(ckpt: Path) -> int:
    name = ckpt.parent.name  # fold0
    if not name.startswith("fold"):
        raise SystemExit(f"unexpected checkpoint path {ckpt}")
    return int(name.replace("fold", ""))


def _recipe_slice(cfg: dict) -> dict:
    mc = cfg.get("model", cfg)
    return {
        "img_size": int(cfg.get("img_size", mc.get("img_size", 224))),
        "stream_mode": str(mc.get("stream_mode", cfg.get("stream_mode", "center"))),
        "n_aux": int(mc.get("n_aux", 0)),
        "spatial_aux": bool(mc.get("spatial_aux", cfg.get("spatial_aux", False))),
    }


def _require_native_299(data_root: Path) -> None:
    """Refuse if img_size=299 but files under data_root are not native 299×299."""
    probe = None
    for split in ("test", "train", "val"):
        for grade in KL_GRADES:
            d = data_root / split / str(grade)
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG_EXTS and not f.name.lower().startswith("gp_"):
                    probe = f
                    break
            if probe is not None:
                break
        if probe is not None:
            break
    if probe is None:
        raise SystemExit(f"no images found under {data_root} to verify native 299px")
    size = Image.open(probe).size
    if size != (299, 299):
        raise SystemExit(
            f"img_size=299 requires --data-root with native kneeKL299 (299×299) images; "
            f"{probe} is {size[0]}×{size[1]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--ensemble", choices=ENSEMBLE_MODES, default="logits",
                    help="average fold logits (default) or class probabilities before decode")
    ap.add_argument(
        "--decode",
        choices=DECODE_MODES,
        default="native",
        help="native cascade decode, or expected-grade cutpoints fit on OOF then applied to test",
    )
    ap.add_argument("--i-understand-this-is-the-final-test", action="store_true")
    args = ap.parse_args()
    if not args.i_understand_this_is_the_final_test:
        sys.exit("Refusing: the test split is locked. Pick the model from the OOF report first.")

    run_dir = args.runs_dir / args.run_name
    ckpts = sorted(run_dir.glob("fold*/best.pth"), key=_fold_index)
    if not ckpts:
        sys.exit(f"no best.pth under {run_dir}")
    fold_ids = [_fold_index(c) for c in ckpts]
    if fold_ids != [0, 1, 2, 3, 4]:
        sys.exit(f"need exactly folds 0-4 with best.pth; found {fold_ids}")

    recipes = []
    for ck in ckpts:
        cfg_path = ck.parent / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
        recipes.append(_recipe_slice(cfg))
    if any(r != recipes[0] for r in recipes[1:]):
        sys.exit(f"fold configs differ on {_CFG_KEYS}: {recipes}")

    suffix = "" if args.decode == "native" else f"_{args.decode}"
    out_file = PROJECT_ROOT / "reports" / "pipeline" / f"final_test_{args.run_name}{suffix}.json"
    if out_file.exists():
        sys.exit(f"{out_file} exists — the final test was already run for this configuration.")

    cuts = None
    if args.decode == "expected_cutpoints":
        cuts = _fit_oof_cutpoints(run_dir)
        print(f"OOF-fit cutpoints: {np.asarray(cuts).round(4).tolist()}")

    recipe = recipes[0]
    if recipe["img_size"] == 299:
        _require_native_299(args.data_root)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with_side = recipe["stream_mode"] == "medlat" or bool(recipe.get("spatial_aux"))
    _, eval_tf = build_transforms(recipe["img_size"], horizontal_flip=True)
    ds = TestSideDataset(args.data_root, eval_tf, with_side=with_side)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    patients = np.array([patient_id(fpath.stem) for fpath, _ in ds.samples])

    raws, probs_list, y = [], [], None
    model = None
    for ck in ckpts:
        c = torch.load(ck, map_location=dev, weights_only=False)
        mc = c["config"].get("model", c["config"])
        model = OrdinalNet.from_config(mc, pretrained=False).to(dev)
        model.load_state_dict(c["model_state_dict"], strict=True)
        pout = predict(model, loader, dev, amp=dev.type == "cuda")
        raw, y = pout[0], pout[1]  # ignore aux_probs if present; locked test uses KL head
        raws.append(raw)
        probs_list.append(model.to_probs(raw))
        print(f"  {ck.parent.name}: done")

    if args.ensemble == "logits":
        blended = torch.stack(raws).mean(0)
        probs = model.to_probs(blended).numpy()
        pred_native = model.decode(blended).numpy()
    else:
        probs = torch.stack(probs_list).mean(0).numpy()
        pred_native = probs.argmax(axis=1)

    if args.decode == "expected_cutpoints":
        pred = apply_cutpoints(expected_grade(probs), cuts)
    else:
        pred = pred_native

    m = metric_bundle(y, pred)
    result = {
        "run": args.run_name,
        "n_models": len(ckpts),
        "n_test": int(len(y)),
        "n_gp_excluded": int(len(ds.base.samples) - len(ds.samples)),
        "ensemble": args.ensemble,
        "decode": args.decode,
        "oof_cutpoints": None if cuts is None else np.asarray(cuts, dtype=float).tolist(),
        "recipe": recipe,
        "data_root": str(args.data_root),
        "metrics": m,
        "ci": bootstrap_ci(y, pred, patients=patients),
        "bootstrap": "patient",
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: m[k] for k in ("qwk", "macro_f1", "accuracy", "kl0_to_kl1", "kl2_to_kl1")}, indent=2))
    print(f"ensemble={args.ensemble} decode={args.decode} → {out_file}")


if __name__ == "__main__":
    main()
