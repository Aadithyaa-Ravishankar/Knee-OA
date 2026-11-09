"""
stage8_gradcam_check.py
========================
Sanity checks for dual-stream Grad-CAM on the Pipeline OrdinalNet champion.

Checks (validation images):
1. Joint-band energy (band_mass) on combined / global / local CAMs.
2. Weight-randomization (Adebayo-style): scrambling last trunk block + KL head
   must change the CAM (mean correlation < 0.5).

Usage:
  python src/training/stage8_gradcam_check.py [--n 16] [--fold 0]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.gradcam_dual import (
    OrdinalNetAdapter,
    band_mass,
    dual_gradcam,
    overlay_cam,
    tensor_to_gray_u8,
)
from src.models.ordinal_net import OrdinalNet
from src.preprocessing.dataset import KneeOADataset
from src.training.stage4_cv_train import build_transforms
from src.training.stage8_gradcam import load_fold_model

MEAN = [0.578707] * 3
STD = [0.145419] * 3
OUT = PROJECT_ROOT / "reports" / "pipeline" / "gradcam_check"


def _randomize_head_and_last_block(model: OrdinalNet) -> OrdinalNet:
    """Copy model; re-init last ResNet layer (or last ConvNeXt/EffNet stage) + KL/aux heads."""
    rand = copy.deepcopy(model)
    torch.manual_seed(0)
    body = rand.body
    # ResNet Sequential: indices 0..7 ≈ conv1..layer4; layer4 is often index 7
    last = None
    if isinstance(body, torch.nn.Sequential) and len(body) >= 1:
        last = body[-1]
    else:
        # ConvNeXt / EffNet: features is Sequential of stages
        children = list(body.children())
        last = children[-1] if children else body
    for m in last.modules():
        if hasattr(m, "reset_parameters"):
            try:
                m.reset_parameters()
            except Exception:
                pass
    if hasattr(rand.fc, "reset_parameters"):
        rand.fc.reset_parameters()
    elif hasattr(rand.fc, "fc") and hasattr(rand.fc.fc, "reset_parameters"):
        rand.fc.fc.reset_parameters()
        if hasattr(rand.fc, "bias"):
            with torch.no_grad():
                rand.fc.bias.normal_(0, 0.02)
    if rand.aux_fc is not None and hasattr(rand.aux_fc, "reset_parameters"):
        rand.aux_fc.reset_parameters()
    rand.eval()
    return rand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--run-name", default="r50_dual_cascade_sqrt_aux")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = args.runs_dir / args.run_name / f"fold{args.fold}" / "best.pth"
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    cfg = json.loads((ckpt.parent / "config.json").read_text(encoding="utf-8"))
    img_size = int(cfg.get("img_size", 224))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fold_model(ckpt, dev)
    adapter = OrdinalNetAdapter(model)
    rand = _randomize_head_and_last_block(model)
    r_adapter = OrdinalNetAdapter(rand)

    _, eval_tf = build_transforms(img_size, horizontal_flip=False)
    ds = KneeOADataset(args.data_root, "val", transform=None)
    samples = [(p, y) for p, y in ds.samples if not p.name.lower().startswith("gp_")]
    rng = np.random.default_rng(args.seed)
    by = {k: [s for s in samples if s[1] == k] for k in range(5)}
    per = max(1, args.n // 5)
    picked = []
    for k in range(5):
        pool = by[k]
        if not pool:
            continue
        take = min(per, len(pool))
        picked.extend(pool[i] for i in rng.choice(len(pool), size=take, replace=False))

    OUT.mkdir(parents=True, exist_ok=True)
    rows, tiles, rand_corr = [], [], []
    for fpath, y in picked:
        from PIL import Image

        x = eval_tf(Image.open(fpath).convert("L")).unsqueeze(0).to(dev)
        cam = dual_gradcam(adapter, x)
        cam_r = dual_gradcam(r_adapter, x)
        a, b = cam["combined"], cam_r["combined"]
        if a.std() == 0 or b.std() == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        rand_corr.append(corr)

        row = {
            "image": fpath.name,
            "true": int(y),
            "pred": cam["pred_class"],
            "band_mass_combined": band_mass(cam["combined"]),
            "band_mass_global": band_mass(cam["streams"][0]),
            "band_mass_local": band_mass(cam["streams"][1]) if len(cam["streams"]) > 1 else None,
            "randomization_corr": corr,
        }
        rows.append(row)

        gray = tensor_to_gray_u8(x, MEAN, STD)
        tile = np.concatenate(
            [
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                overlay_cam(gray, cam["combined"]),
                overlay_cam(gray, cam["streams"][0]),
                overlay_cam(gray, cam["streams"][1]) if len(cam["streams"]) > 1 else overlay_cam(gray, cam["combined"]),
            ],
            axis=1,
        )
        cv2.putText(
            tile, f"T{y} P{cam['pred_class']}", (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        tiles.append(tile)

    header = np.zeros((22, tiles[0].shape[1], 3), np.uint8)
    for j, name in enumerate(["original", "combined", "global", "local"]):
        cv2.putText(header, name, (j * gray.shape[1] + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(OUT / "champion_gradcam_contact_sheet.png"), np.concatenate([header] + tiles, axis=0))

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "run_name": args.run_name,
        "fold": args.fold,
        "n_images": len(rows),
        "mean_band_mass_combined": mean("band_mass_combined"),
        "mean_band_mass_global": mean("band_mass_global"),
        "mean_band_mass_local": mean("band_mass_local"),
        "band_mass_uniform_baseline": band_mass(np.ones((224, 224), dtype=np.float32)),
        "randomization_test_mean_corr": float(np.mean(rand_corr)),
        "randomization_test_pass": bool(np.mean(rand_corr) < 0.5),
        "rows": rows,
    }
    (OUT / "gradcam_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"Contact sheet → {OUT / 'champion_gradcam_contact_sheet.png'}")


if __name__ == "__main__":
    main()
