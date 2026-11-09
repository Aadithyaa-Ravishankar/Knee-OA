"""
stage10_uncertainty.py
======================
Stage 10 — Uncertainty quantification & probability calibration (no retrain).

- 30-pass MC Dropout predictive distribution (dropout active, BN frozen)
- Predictive entropy
- Expected Calibration Error (ECE) + reliability diagram
- Hard grades via expected-grade cutpoints (champion decode) when requested

Example:
  python src/training/stage10_uncertainty.py \\
      --run-name r50_dual_cascade_sqrt_aux_attn_spatial \\
      --decode expected_cutpoints --n-passes 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.decoding import apply_cutpoints, expected_grade
from src.evaluation.uncertainty import (
    enable_mc_dropout,
    reliability_diagram_png,
    sanitize_probs,
    summarize_uncertainty,
)
from src.models.ordinal_net import OrdinalNet
from src.training.stage7_final_test import TestSideDataset, _fit_oof_cutpoints, _fold_index
from src.training.stage4_cv_train import build_transforms


def _parse_folds(text: str) -> list[int]:
    text = text.strip().lower()
    if text in ("all", "*"):
        return [0, 1, 2, 3, 4]
    return [int(x) for x in text.split(",") if x.strip() != ""]


def _forward_probs(model: OrdinalNet, x: torch.Tensor, side: torch.Tensor | None, amp: bool) -> torch.Tensor:
    """Class probs; fall back to fp32 if AMP yields non-finite values."""
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
        raw = model(x, side)
        probs = model.to_probs(raw).float()
    if amp and (not torch.isfinite(probs).all()):
        raw = model(x, side)
        probs = model.to_probs(raw).float()
    return probs


@torch.no_grad()
def mc_fold_probs(
    model: OrdinalNet,
    loader: DataLoader,
    dev: torch.device,
    n_passes: int,
    amp: bool,
    with_side: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean MC class probs over the loader. Returns (probs N×5, labels N)."""
    enable_mc_dropout(model)
    acc_sum = None
    y_all = []
    for batch in loader:
        x = batch[0].to(dev, non_blocking=True)
        y = batch[1]
        side = batch[-1].to(dev, non_blocking=True) if with_side else None
        batch_acc = None
        for _ in range(int(n_passes)):
            probs = _forward_probs(model, x, side, amp)
            batch_acc = probs if batch_acc is None else batch_acc + probs
        mean = (batch_acc / float(n_passes)).cpu()
        acc_sum = mean if acc_sum is None else torch.cat([acc_sum, mean], dim=0)
        y_all.append(y)
    model.eval()
    return sanitize_probs(acc_sum.numpy()), torch.cat(y_all).numpy()


@torch.no_grad()
def eval_fold_probs(
    model: OrdinalNet,
    loader: DataLoader,
    dev: torch.device,
    amp: bool,
    with_side: bool,
) -> np.ndarray:
    """Deterministic eval-mode class probs (for optional comparison)."""
    model.eval()
    outs = []
    for batch in loader:
        x = batch[0].to(dev, non_blocking=True)
        side = batch[-1].to(dev, non_blocking=True) if with_side else None
        outs.append(_forward_probs(model, x, side, amp).cpu())
    return sanitize_probs(torch.cat(outs).numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="r50_dual_cascade_sqrt_aux_attn_spatial")
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--n-passes", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--decode",
        choices=("native", "expected_cutpoints"),
        default="expected_cutpoints",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: reports/pipeline/uncertainty_<run>",
    )
    args = ap.parse_args()

    run_dir = args.runs_dir / args.run_name
    out_dir = args.out or (PROJECT_ROOT / "reports" / "pipeline" / f"uncertainty_{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = _parse_folds(args.folds)
    ckpts = [run_dir / f"fold{f}" / "best.pth" for f in folds]
    for ck in ckpts:
        if not ck.is_file():
            raise SystemExit(f"missing {ck}")

    cuts = _fit_oof_cutpoints(run_dir) if args.decode == "expected_cutpoints" else None
    cfg = json.loads((ckpts[0].parent / "config.json").read_text(encoding="utf-8"))
    mc = {**cfg, **cfg.get("model", {})}
    img_size = int(cfg.get("img_size", mc.get("img_size", 224)))
    with_side = str(mc.get("stream_mode", "center")) == "medlat" or bool(mc.get("spatial_aux", False))
    _, eval_tf = build_transforms(img_size, horizontal_flip=True)

    ds = TestSideDataset(args.data_root, eval_tf, with_side=with_side)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"

    print(
        f"Stage 10: n_test={len(ds)}  folds={folds}  N={args.n_passes}  decode={args.decode}",
        flush=True,
    )

    mc_stack = []
    det_stack = []
    y = None
    for ck in ckpts:
        c = torch.load(ck, map_location=dev, weights_only=False)
        mcfg = c["config"].get("model", c["config"])
        model = OrdinalNet.from_config(mcfg, pretrained=False).to(dev)
        model.load_state_dict(c["model_state_dict"], strict=True)
        print(f"  MC fold {_fold_index(ck)} …", flush=True)
        probs_mc, y = mc_fold_probs(model, loader, dev, args.n_passes, amp, with_side)
        probs_det = eval_fold_probs(model, loader, dev, amp, with_side)
        mc_stack.append(probs_mc)
        det_stack.append(probs_det)

    mc_probs = np.mean(np.stack(mc_stack, axis=0), axis=0)
    det_probs = np.mean(np.stack(det_stack, axis=0), axis=0)

    def decode(probs: np.ndarray) -> np.ndarray:
        if cuts is not None:
            return apply_cutpoints(expected_grade(probs), cuts).astype(np.int64)
        # native: argmax of class probs (matches OrdinalNet.to_probs decode path closely)
        return probs.argmax(axis=1).astype(np.int64)

    pred_mc = decode(mc_probs)
    pred_det = decode(det_probs)

    summary_mc = summarize_uncertainty(mc_probs, y, pred_mc, n_bins=args.n_bins)
    summary_det = summarize_uncertainty(det_probs, y, pred_det, n_bins=args.n_bins)

    reliability_diagram_png(
        summary_mc["ece_report"],
        out_dir / "reliability_mc.png",
        title=f"MC Dropout reliability ({args.run_name})",
    )
    reliability_diagram_png(
        summary_det["ece_report"],
        out_dir / "reliability_deterministic.png",
        title=f"Deterministic reliability ({args.run_name})",
    )

    # Per-image rows (compact)
    from src.evaluation.uncertainty import predictive_entropy

    ent = predictive_entropy(mc_probs)
    conf = mc_probs[np.arange(len(pred_mc)), pred_mc]
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(y)):
            f.write(
                json.dumps(
                    {
                        "true": int(y[i]),
                        "pred_mc": int(pred_mc[i]),
                        "pred_det": int(pred_det[i]),
                        "entropy": float(ent[i]),
                        "confidence": float(conf[i]),
                        "correct_mc": bool(pred_mc[i] == y[i]),
                    }
                )
                + "\n"
            )

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    meta = {
        "stage": 10,
        "run_name": args.run_name,
        "folds": folds,
        "n_passes": args.n_passes,
        "n_bins": args.n_bins,
        "decode": args.decode,
        "oof_cutpoints": None if cuts is None else [float(c) for c in np.asarray(cuts)],
        "mc": {k: v for k, v in summary_mc.items() if k != "ece_report"},
        "deterministic": {k: v for k, v in summary_det.items() if k != "ece_report"},
        "ece_bins_mc": summary_mc["ece_report"]["bins"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(meta), indent=2), encoding="utf-8"
    )

    def _f(v, nd=4):
        return "n/a" if v is None else f"{v:.{nd}f}"

    lines = [
        "# Stage 10 — Uncertainty & Calibration",
        "",
        f"- run: `{args.run_name}`  folds: `{folds}`  MC passes: **{args.n_passes}**",
        f"- decode: `{args.decode}`  n_test: **{summary_mc['n']}**",
        "",
        "## MC Dropout (primary)",
        "",
        f"- **ECE:** {_f(summary_mc['ece'])}  ({args.n_bins} bins)",
        f"- **mean predictive entropy:** {_f(summary_mc['mean_entropy'])}",
        f"- **mean confidence (P[pred]):** {_f(summary_mc['mean_confidence'])}",
        f"- **accuracy:** {_f(summary_mc['accuracy'])}",
        f"- entropy by true KL: {summary_mc['entropy_by_true_kl']}",
        "",
        "## Deterministic ensemble (reference)",
        "",
        f"- **ECE:** {_f(summary_det['ece'])}",
        f"- **mean entropy:** {_f(summary_det['mean_entropy'])}",
        f"- **accuracy:** {_f(summary_det['accuracy'])}",
        "",
        "## Artifacts",
        "",
        "- `reliability_mc.png` / `reliability_deterministic.png`",
        "- `summary.json`, `rows.jsonl`",
        "",
        "Dropout layers are forced to train mode; BatchNorm stays in eval. "
        "Fold probabilities are averaged, then hard grades use the Stage 7 cutpoints when selected.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_jsonable({"mc": meta["mc"], "deterministic": meta["deterministic"]}), indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
