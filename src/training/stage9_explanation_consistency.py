"""
stage9_explanation_consistency.py
=================================
Stage 9 — Explanation Consistency on the locked champion (no retrain).

Computes population:
  - AFS (= band_mass on Grad-CAM)
  - ESS (mean SSIM of CAM under brightness ±5% and Gaussian noise σ=0.01)
  - Grade flip rate under the same perturbations

Uses fold-0 CAM by default (matches Stage 8); hard grades from 5-fold ensemble
+ expected-grade cutpoints when --decode expected_cutpoints.

Example:
  python src/training/stage9_explanation_consistency.py \\
      --run-name r50_dual_cascade_sqrt_aux_attn_spatial \\
      --n 200 --decode expected_cutpoints
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

from src.evaluation.consistency import (
    default_perturbations,
    explanation_stability,
    grade_flipped,
    summarize_consistency,
)
from src.evaluation.gradcam_dual import OrdinalNetAdapter, band_mass, dual_gradcam
from src.training.stage8_gradcam import (
    collect_split_samples,
    ensemble_predict,
    fit_run_cutpoints,
    load_fold_model,
    stratified_pick,
)
from src.training.stage4_cv_train import build_transforms, side_code


def _parse_folds(text: str) -> list[int]:
    text = text.strip().lower()
    if text in ("all", "*"):
        return [0, 1, 2, 3, 4]
    return [int(x) for x in text.split(",") if x.strip() != ""]


def _load_models(run_dir: Path, folds: list[int], dev: torch.device):
    models = []
    for f in folds:
        ckpt = run_dir / f"fold{f}" / "best.pth"
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        models.append(load_fold_model(ckpt, dev))
    return models


def _tensor_for_path(fpath: Path, eval_tf, with_side: bool, dev: torch.device):
    img = Image.open(fpath).convert("L")
    x = eval_tf(img).unsqueeze(0).to(dev)
    side = None
    if with_side:
        side = torch.tensor([side_code(fpath.name)], device=dev, dtype=torch.long)
    return x, side


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="r50_dual_cascade_sqrt_aux_attn_spatial")
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--n", type=int, default=200, help="stratified sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred-folds", default="all", help="folds for ensemble grade (default all)")
    ap.add_argument("--cam-fold", type=int, default=0, help="single fold for Grad-CAM / ESS")
    ap.add_argument(
        "--decode",
        choices=("native", "expected_cutpoints"),
        default="expected_cutpoints",
    )
    ap.add_argument("--noise-sigma", type=float, default=0.01)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: reports/pipeline/explanation_consistency_<run>",
    )
    args = ap.parse_args()

    run_dir = args.runs_dir / args.run_name
    out_dir = args.out or (
        PROJECT_ROOT / "reports" / "pipeline" / f"explanation_consistency_{args.run_name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_folds = _parse_folds(args.pred_folds)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_models(run_dir, pred_folds, dev)
    cam_model = load_fold_model(run_dir / f"fold{args.cam_fold}" / "best.pth", dev)

    cfg_path = run_dir / f"fold{args.cam_fold}" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    mc = {**cfg, **cfg.get("model", {})}
    img_size = int(cfg.get("img_size", mc.get("img_size", 224)))
    with_side = str(mc.get("stream_mode", "center")) == "medlat" or bool(mc.get("spatial_aux", False))
    _, eval_tf = build_transforms(img_size, horizontal_flip=True)

    cuts = fit_run_cutpoints(run_dir) if args.decode == "expected_cutpoints" else None

    samples = collect_split_samples(args.data_root, args.split)
    picked = samples if args.n <= 0 else stratified_pick(samples, args.n, seed=args.seed)
    print(f"Stage 9: n={len(picked)}/{len(samples)}  decode={args.decode}  cam_fold={args.cam_fold}", flush=True)

    rows = []
    for i, (fpath, true_y) in enumerate(picked, 1):
        x, side = _tensor_for_path(fpath, eval_tf, with_side, dev)
        pred0, probs0 = ensemble_predict(models, x, cuts=cuts)

        adapter = OrdinalNetAdapter(cam_model, side=side)
        cam0 = dual_gradcam(adapter, x, target_class=pred0)
        afs_c = band_mass(cam0["combined"])
        afs_g = band_mass(cam0["streams"][0])
        afs_l = band_mass(cam0["streams"][1]) if len(cam0["streams"]) > 1 else None

        pert_cams = []
        pert_preds = []
        pert_names = []
        for name, xp in default_perturbations(x, noise_sigma=args.noise_sigma, seed=args.seed + i):
            pred_p, _ = ensemble_predict(models, xp, cuts=cuts)
            cam_p = dual_gradcam(adapter, xp, target_class=pred0)
            pert_cams.append(cam_p["combined"])
            pert_preds.append(int(pred_p))
            pert_names.append(name)

        ess = explanation_stability(cam0["combined"], pert_cams)
        flipped = grade_flipped(pred0, pert_preds)
        rows.append(
            {
                "path": str(fpath.relative_to(args.data_root)) if fpath.is_relative_to(args.data_root) else str(fpath),
                "true": int(true_y),
                "pred": int(pred0),
                "probs": [float(p) for p in probs0],
                "afs_combined": float(afs_c),
                "afs_global": float(afs_g),
                "afs_local": None if afs_l is None else float(afs_l),
                "ess": float(ess),
                "grade_flipped": bool(flipped),
                "pert_preds": {n: p for n, p in zip(pert_names, pert_preds)},
            }
        )
        if i == 1 or i % 25 == 0 or i == len(picked):
            print(f"  {i}/{len(picked)}  running_mean_ess={np.mean([r['ess'] for r in rows]):.3f}  "
                  f"flip_rate={np.mean([r['grade_flipped'] for r in rows]):.3f}", flush=True)

    summary = summarize_consistency(rows)
    meta = {
        "stage": 9,
        "run_name": args.run_name,
        "split": args.split,
        "decode": args.decode,
        "pred_folds": pred_folds,
        "cam_fold": args.cam_fold,
        "n_requested": args.n,
        "perturbations": ["brightness_0.95", "brightness_1.05", f"noise_{args.noise_sigma}"],
        "summary": summary,
    }

    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    lines = [
        "# Stage 9 — Explanation Consistency",
        "",
        f"- run: `{args.run_name}`  split: `{args.split}`  decode: `{args.decode}`",
        f"- n: **{summary['n']}**  CAM fold: `{args.cam_fold}`  pred folds: `{pred_folds}`",
        f"- perturbations: brightness ±5%, Gaussian noise σ={args.noise_sigma}",
        "",
        "## Population metrics",
        "",
        (
            f"- **mean AFS (combined / global / local):** "
            f"{summary['mean_afs_combined']:.3f} / {summary['mean_afs_global']:.3f} / "
            f"{'None' if summary['mean_afs_local'] is None else format(summary['mean_afs_local'], '.3f')} "
            f"(uniform baseline {summary['band_mass_uniform_baseline']:.3f})"
        ),
        (
            f"- **frac AFS > 0.5:** {summary['frac_afs_gt_0_5']:.3f}  "
            f"(target mean AFS > 0.5 → {'PASS' if summary['pass_afs'] else 'FAIL'})"
        ),
        (
            f"- **mean ESS (SSIM):** {summary['mean_ess']:.3f}  "
            f"(frac > 0.85: {summary['frac_ess_gt_0_85']:.3f}; "
            f"target mean > 0.85 → {'PASS' if summary['pass_ess'] else 'FAIL'})"
        ),
        (
            f"- **grade flip rate:** {summary['grade_flip_rate']:.3f}  "
            f"(target < 0.05 → {'PASS' if summary['pass_grade_flip'] else 'FAIL'})"
        ),
        "",
        "AFS uses the same joint-band energy as Stage 8 `band_mass`. "
        "ESS is mean SSIM between original and perturbed combined CAMs. "
        "Grade flip counts any hard-grade change under the same perturbations.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
