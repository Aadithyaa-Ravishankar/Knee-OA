"""
stage6_screen_joint_focus.py
=============================
Compare Stage-19 screen runs on (mean OOF/val QWK) + Grad-CAM global joint-band mass.

Does not retrain. Point --runs-dir at downloaded Kaggle weights (and the locked
champion baseline). Writes a ranking table for the win rule:

  keep champion unless a candidate has higher mean fold QWK AND higher mean
  global band mass on a fixed test sample (joint-focus gate).

Example:
  python src/training/stage6_screen_joint_focus.py \\
      --runs-dir outputs/cv/runs \\
      --candidates r50_dual_cascade_sqrt_aux \\
                   r50_dual_cascade_sqrt_aux_localheavy \\
                   r50_dual_cascade_sqrt_aux_edgemask \\
                   r50_dual_cascade_sqrt_aux_localheavy_edgemask \\
      --out reports/pipeline/screen_joint_focus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.attention_priors import border_mass_numpy
from src.evaluation.gradcam_dual import OrdinalNetAdapter, band_mass, dual_gradcam
from src.training.stage4_cv_train import build_transforms
from src.training.stage8_gradcam import collect_split_samples, load_fold_model, stratified_pick


def fold_qwk(run_dir: Path) -> dict:
    qwks = []
    for done in sorted(run_dir.glob("fold*/DONE")):
        data = json.loads(done.read_text(encoding="utf-8"))
        if "qwk" in data:
            qwks.append(float(data["qwk"]))
    return {
        "n_folds_done": len(qwks),
        "mean_qwk": float(np.mean(qwks)) if qwks else None,
        "fold_qwks": qwks,
    }


def mean_global_band(
    run_dir: Path,
    samples: list[tuple[Path, int]],
    cam_fold: int,
    dev: torch.device,
) -> dict:
    ckpt = run_dir / f"fold{cam_fold}" / "best.pth"
    if not ckpt.is_file():
        return {"mean_global_band_mass": None, "n_cam": 0, "error": f"missing {ckpt}"}
    model = load_fold_model(ckpt, dev)
    img_size = int(
        json.loads((ckpt.parent / "config.json").read_text(encoding="utf-8")).get("img_size", 224)
    )
    _, eval_tf = build_transforms(img_size, horizontal_flip=False)
    adapter = OrdinalNetAdapter(model)
    bands, borders, locals_ = [], [], []
    for fpath, _y in samples:
        x = eval_tf(Image.open(fpath).convert("L")).unsqueeze(0).to(dev)
        x.requires_grad_(True)
        out = dual_gradcam(adapter, x, target_class=None)
        g = out["streams"][0]
        bands.append(band_mass(g))
        borders.append(border_mass_numpy(g))
        if len(out["streams"]) > 1:
            locals_.append(band_mass(out["streams"][1]))
    return {
        "mean_global_band_mass": float(np.mean(bands)) if bands else None,
        "mean_global_border_mass": float(np.mean(borders)) if borders else None,
        "mean_local_band_mass": float(np.mean(locals_)) if locals_ else None,
        "n_cam": len(bands),
        "cam_fold": cam_fold,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--baseline", default="r50_dual_cascade_sqrt_aux")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-cam", type=int, default=40, help="stratified images for band-mass probe")
    ap.add_argument("--cam-fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "pipeline" / "screen_joint_focus.json")
    args = ap.parse_args(argv)

    samples = stratified_pick(collect_split_samples(args.data_root, args.split), args.n_cam, args.seed)
    baseline = band_mass(np.ones((224, 224), dtype=np.float32))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for name in args.candidates:
        run_dir = args.runs_dir / name
        if not run_dir.is_dir():
            rows.append({"run": name, "error": f"missing {run_dir}"})
            continue
        q = fold_qwk(run_dir)
        b = mean_global_band(run_dir, samples, args.cam_fold, dev)
        rows.append({"run": name, **q, **b})

    base = next((r for r in rows if r["run"] == args.baseline and r.get("mean_qwk") is not None), None)
    for r in rows:
        if base is None or r.get("mean_qwk") is None or r.get("mean_global_band_mass") is None:
            r["beats_baseline"] = False
            continue
        # Win: QWK not worse (allow tiny float noise) AND lower frame-border CAM mass
        # AND joint-band mass at least as high as baseline.
        qwk_ok = r["mean_qwk"] >= base["mean_qwk"] - 0.002
        border_ok = (
            r.get("mean_global_border_mass") is not None
            and base.get("mean_global_border_mass") is not None
            and r["mean_global_border_mass"] < base["mean_global_border_mass"]
        )
        band_ok = r["mean_global_band_mass"] >= base["mean_global_band_mass"]
        r["beats_baseline"] = r["run"] != args.baseline and qwk_ok and border_ok and band_ok

    ranked = sorted(
        [r for r in rows if r.get("mean_qwk") is not None],
        key=lambda r: (r.get("beats_baseline", False), r.get("mean_qwk") or -1),
        reverse=True,
    )
    payload = {
        "baseline": args.baseline,
        "band_mass_uniform_baseline": baseline,
        "n_cam_images": len(samples),
        "win_rule": "mean_qwk > baseline AND mean_global_band_mass > baseline",
        "rows": rows,
        "ranked": [r["run"] for r in ranked],
        "winner": next((r["run"] for r in ranked if r.get("beats_baseline")), args.baseline),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
