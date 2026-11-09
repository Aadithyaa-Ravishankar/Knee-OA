"""
stage5_dump_aux.py
===================
Re-run fold-val inference for an existing aux-head checkpoint and write aux_oof.npz
(no retraining). Uses that fold's VAL rows from folds.csv only.

Outputs in <runs-dir>/<run-name>/fold<k>/aux_oof.npz:
  paths, labels, aux_probs (N, n_aux, 3), probs (KL), pred, fold

Example:
  python src/training/stage5_dump_aux.py --run-name r50_dual_cascade_sqrt_aux \\
      --folds 0,1 --runs-dir outputs/cv/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.protocol import read_folds_csv
from src.models.ordinal_net import OrdinalNet
from src.training.stage4_cv_train import (
    AUX_COLS,
    FoldDataset,
    build_transforms,
    load_medlat_mapping,
    predict,
    remap_rows_to_data_root,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--folds", default="0,1", help="comma-separated fold indices")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--folds-csv", type=Path, default=PROJECT_ROOT / "reports" / "pipeline" / "folds.csv")
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs",
                    help="directory containing <run-name>/foldk/{best.pth,config.json}")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args(argv)


def load_model(fold_dir: Path, dev: torch.device) -> tuple[OrdinalNet, dict]:
    cfg_path = fold_dir / "config.json"
    ckpt_path = fold_dir / "best.pth"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        raise SystemExit(f"need {cfg_path.name} and {ckpt_path.name} in {fold_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    mc = cfg.get("model") or ckpt.get("config") or {}
    n_aux = int(mc.get("n_aux", 0))
    if n_aux <= 0:
        raise SystemExit(f"{fold_dir}: model.n_aux={n_aux}; dump_aux requires aux heads")
    stream_mode = mc.get("stream_mode", cfg.get("stream_mode", "center"))
    med_left = mc.get("medial_is_image_left_for_r")
    if med_left is None and stream_mode == "medlat":
        med_left = load_medlat_mapping(None)
    elif med_left is None:
        med_left = True
    mc_full = {
        **cfg,
        **mc,
        "n_aux": n_aux,
        "stream_mode": stream_mode,
        "medial_is_image_left_for_r": bool(med_left),
        "dual": mc.get("dual", not cfg.get("no_dual", False)),
    }
    model = OrdinalNet.from_config(mc_full, dropout=float(cfg.get("dropout", 0.3)), pretrained=False).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def dump_fold(args, fold: int, rows: list[dict], dev: torch.device) -> Path:
    fold_dir = args.runs_dir / args.run_name / f"fold{fold}"
    model, cfg = load_model(fold_dir, dev)
    img_size = int(cfg.get("img_size", 224))
    stream_mode = getattr(model, "stream_mode", "center")
    use_medlat = stream_mode == "medlat"
    val_rows = [r for r in rows if r["fold"] == fold]
    if not val_rows:
        raise SystemExit(f"no val rows for fold {fold}")
    _, eval_tf = build_transforms(img_size, horizontal_flip=False)
    ds = FoldDataset(val_rows, args.data_root, eval_tf, with_aux=False,
                     with_side=use_medlat, flip_swap_side=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=dev.type == "cuda")
    amp = dev.type == "cuda" and not args.no_amp
    pout = predict(model, loader, dev, amp)
    raw, yv, aux_probs = pout[0], pout[1], pout[2]
    probs = model.to_probs(raw).numpy()
    pred = model.decode(raw).numpy()
    n = len(yv)
    out = fold_dir / "aux_oof.npz"
    np.savez_compressed(
        out,
        paths=np.array([r["path"] for r in val_rows[:n]]),
        labels=yv,
        aux_probs=aux_probs,
        aux_names=np.array(AUX_COLS[: model.n_aux]),
        probs=probs,
        pred=pred,
        fold=np.full(n, fold),
    )
    print(f"fold {fold}: wrote {out} n={n} aux_probs={aux_probs.shape}")
    return out


def main(argv=None):
    args = parse_args(argv)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    rows = remap_rows_to_data_root(read_folds_csv(args.folds_csv), args.data_root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f in folds:
        dump_fold(args, f, rows, dev)


if __name__ == "__main__":
    main()
