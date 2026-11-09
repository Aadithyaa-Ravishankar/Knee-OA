"""
stage4_cv_train.py
===================
Train ONE fold of a Pipeline configuration. Resume-safe: re-running the same
command continues from the last finished epoch (for Kaggle/Colab disconnects).

Data  : patient-grouped folds over Chen train+val (no G&P). Test is never read.
Model : src.models.ordinal_net.OrdinalNet (backbone / dual stream / head configurable)
Recipe: AdamW (backbone LR, head LR x10), linear warmup + cosine, EMA weights,
        mixed precision on CUDA, flip TTA at evaluation, early stopping on fold-val QWK.

Outputs in <out-dir>/<run-name>/fold<k>/:
  last.pth      full training state (for resume)
  best.pth      best EMA weights + config
  history.csv   one row per epoch
  oof.npz       fold-val predictions of best.pth (paths, labels, raw, probs, pred)
  DONE          marker; a finished fold is skipped on re-run

Example:
  python src/training/stage4_cv_train.py --fold 0 --run-name r50_dual_cascade_sqrt \
      --backbone resnet50 --head cascade --sampler sqrt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.attention_priors import attention_prior_loss
from src.evaluation.protocol import metric_bundle, read_folds_csv
from src.models.ordinal_net import BACKBONES, FUSION_MODES, HEADS, STREAM_MODES, OrdinalNet
from src.preprocessing.transforms import get_eval_transforms_3ch, get_train_transforms_3ch

MEAN = [0.578707] * 3
STD = [0.145419] * 3
STAGE6_SCALE = {0: 0.45, 1: 1.35, 2: 0.75, 3: 1.0, 4: 1.15}
PAUSED_EXIT = 75  # stopped early because PIPELINE_DEADLINE (unix time) is near; re-run to resume
AUX_COLS = ("aux_jsn_med", "aux_jsn_lat", "aux_ost_med", "aux_ost_lat")
MEDLAT_MAP = PROJECT_ROOT / "reports" / "pipeline" / "medlat_side_mapping.json"


def aux_vector(row: dict) -> torch.Tensor:
    vals = []
    for col in AUX_COLS:
        text = str(row.get(col, "") or "").strip()
        vals.append(-1 if text == "" or text.lower() == "nan" else int(float(text)))
    return torch.tensor(vals, dtype=torch.long)


def side_code(path: str) -> int:
    """0=L, 1=R from filename stem."""
    stem = Path(path).stem
    if stem[-1:].upper() == "R":
        return 1
    if stem[-1:].upper() == "L":
        return 0
    raise ValueError(f"cannot read L/R side from {path}")


class FoldDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        data_root: Path,
        transform,
        with_aux: bool = False,
        with_side: bool = False,
        flip_swap_side: bool = False,
    ):
        self.rows = rows
        self.root = Path(data_root)
        self.transform = transform
        self.with_aux = with_aux
        self.with_side = with_side
        self.flip_swap_side = flip_swap_side  # HFlip here so R/L stays anatomically correct

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["path"]).convert("L")
        side = side_code(r["path"]) if (self.with_side or self.flip_swap_side) else None
        if self.flip_swap_side and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            side = 1 - side
        out = [self.transform(img), r["label"]]
        if self.with_aux:
            out.append(aux_vector(r))
        if self.with_side:
            out.append(side)
        return tuple(out)


def build_transforms(
    img_size: int,
    horizontal_flip: bool = True,
    aug_interp: str = "nearest",
    random_erasing: bool = True,
):
    train_tf = get_train_transforms_3ch(
        MEAN, STD, horizontal_flip=horizontal_flip,
        aug_interp=aug_interp, random_erasing=random_erasing,
    )
    eval_tf = get_eval_transforms_3ch(MEAN, STD)
    if img_size != 224:
        resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC)
        train_tf = T.Compose([resize, train_tf])
        eval_tf = T.Compose([resize, eval_tf])
    return train_tf, eval_tf


# Recipe fields compared before skipping a DONE fold (must match requested args).
_DONE_COMPARE_KEYS = (
    "backbone", "head", "no_dual", "local_frac", "sampler", "sord_alpha", "img_size",
    "batch_size", "lr", "weight_decay", "dropout", "ema_decay", "aux_weight", "soft_kl12",
    "soft_oa01", "aux_jsn_med_weight",
    "stream_mode", "aug_interp", "no_random_erasing",
    "fusion", "periphery_suppress", "global_roi_frac",
    "spatial_aux", "spatial_aux_global", "attn_band_weight", "attn_border_frac",
    "attn_global_only",
)


def _norm_cfg_val(key: str, val):
    if key == "aux_csv":
        if val in (None, "None", ""):
            return None
        return Path(str(val)).name
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, bool) or val in (None,):
        return val
    if key in ("local_frac", "sord_alpha", "lr", "weight_decay", "dropout", "ema_decay",
               "aux_weight", "soft_kl12", "soft_oa01", "aux_jsn_med_weight", "warmup_epochs",
               "periphery_suppress", "global_roi_frac",
               "attn_band_weight", "attn_border_frac"):
        return float(val)
    if key in ("img_size", "batch_size", "epochs", "patience", "seed"):
        return int(val)
    if key in (
        "no_dual", "no_pretrained", "no_amp", "no_random_erasing",
        "spatial_aux", "spatial_aux_global", "attn_global_only",
    ):
        return bool(val)
    return val


def assert_done_matches_args(fold_dir: Path, args) -> None:
    """If DONE exists, skip only when config.json matches the requested recipe; else refuse."""
    cfg_path = fold_dir / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"[refuse] {fold_dir} has DONE but no config.json; delete DONE to retrain")
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    requested = {k: getattr(args, k) for k in _DONE_COMPARE_KEYS}
    requested["aux_csv"] = args.aux_csv
    mismatches = []
    for key, req in requested.items():
        if key not in saved and key in (
            "soft_kl12", "soft_oa01", "aux_jsn_med_weight", "aug_interp", "no_random_erasing",
            "stream_mode",
            "fusion", "periphery_suppress", "global_roi_frac",
            "spatial_aux", "spatial_aux_global", "attn_band_weight", "attn_border_frac",
            "attn_global_only",
        ):
            # older runs omit new keys; treat as defaults
            if key == "soft_kl12" and float(req) == 0.0:
                continue
            if key == "soft_oa01" and float(req) == 0.0:
                continue
            if key == "aux_jsn_med_weight" and float(req) == 1.0:
                continue
            if key == "aug_interp" and req == "nearest":
                continue
            if key == "no_random_erasing" and req is False:
                continue
            if key == "stream_mode" and req == "center":
                continue
            if key == "fusion" and req == "concat":
                continue
            if key == "periphery_suppress" and float(req) == 0.0:
                continue
            if key == "global_roi_frac" and float(req) == 0.85:
                continue
            if key == "spatial_aux" and req is False:
                continue
            if key == "spatial_aux_global" and req is False:
                continue
            if key == "attn_band_weight" and float(req) == 0.0:
                continue
            if key == "attn_border_frac" and float(req) == 0.08:
                continue
            if key == "attn_global_only" and req is False:
                continue
        sv = _norm_cfg_val(key, saved.get(key))
        rv = _norm_cfg_val(key, req)
        if sv != rv:
            mismatches.append(f"{key}: saved={sv!r} requested={rv!r}")
    if mismatches:
        raise SystemExit(
            f"[refuse] {fold_dir} is DONE but config differs from this command:\n  - "
            + "\n  - ".join(mismatches)
            + "\nDelete that fold's DONE (and preferably the fold dir) to retrain, "
            "or restore matching args."
        )


def load_medlat_mapping(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    if not MEDLAT_MAP.is_file():
        raise SystemExit(f"missing {MEDLAT_MAP}; run src/preprocessing/verify_medlat_side.py first")
    data = json.loads(MEDLAT_MAP.read_text(encoding="utf-8"))
    if data.get("ambiguous"):
        raise SystemExit(f"{MEDLAT_MAP} marked ambiguous; do not train medlat until resolved")
    return bool(data["medial_is_image_left_for_r"])


def sample_weights(labels: list[int], mode: str) -> torch.Tensor | None:
    counts = np.bincount(labels, minlength=5).astype(float)
    if mode == "natural":
        return None
    if mode == "sqrt":
        per_class = 1.0 / np.sqrt(np.maximum(counts, 1))
    elif mode == "stage6":
        per_class = np.array([STAGE6_SCALE[k] for k in range(5)]) / np.maximum(counts, 1)
    else:
        raise ValueError(mode)
    return torch.tensor([per_class[y] for y in labels], dtype=torch.double)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def remap_rows_to_data_root(rows: list[dict], data_root: Path) -> list[dict]:
    """Match folds.csv paths to files under data_root by filename (train/val only)."""
    index: dict[str, str] = {}
    for split in ("train", "val"):
        for grade in range(5):
            d = Path(data_root) / split / str(grade)
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"} and not f.name.lower().startswith("gp_"):
                    index[f.name] = f"{split}/{grade}/{f.name}"
    out, missing = [], []
    for r in rows:
        name = Path(r["path"]).name
        hit = index.get(name)
        if hit is None:
            missing.append(r["path"])
            continue
        nr = dict(r)
        nr["path"] = hit
        out.append(nr)
    print(f"data-root map: {len(out)}/{len(rows)} fold rows matched under {data_root}")
    if missing:
        raise SystemExit(f"{len(missing)} fold images missing under {data_root}, e.g. {missing[:3]}")
    return out


@torch.no_grad()
def predict(model: OrdinalNet, loader, dev, amp: bool, max_batches: int | None = None):
    """Returns (raw_kl, labels) or (raw_kl, labels, aux_probs) when model.n_aux > 0.

    aux_probs : (N, n_aux, 3) sigmoid of flip-TTA-averaged aux logits.
    """
    model.eval()
    medlat = getattr(model, "stream_mode", "center") == "medlat"
    spatial_aux = bool(getattr(model, "spatial_aux", False))
    need_side = medlat or spatial_aux
    want_aux = getattr(model, "n_aux", 0) > 0 and (model.aux_fc is not None or spatial_aux)
    raws, auxs, ys = [], [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = batch[0].to(dev, non_blocking=True), batch[1]
        side = batch[-1].to(dev, non_blocking=True) if need_side else None

        def _one(inp, sd):
            if want_aux:
                rk, ra = model.forward_with_aux(inp, sd)
                return rk, ra
            return model(inp, sd), None

        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            if need_side:
                r0, a0 = _one(x, side)
                r1, a1 = _one(torch.flip(x, dims=[3]), 1 - side)
            else:
                r0, a0 = _one(x, None)
                r1, a1 = _one(torch.flip(x, dims=[3]), None)
            r = (r0.float() + r1.float()) / 2
            a = None if a0 is None else (a0.float() + a1.float()) / 2
        if amp and not torch.isfinite(r).all():
            if need_side:
                r0, a0 = _one(x, side)
                r1, a1 = _one(torch.flip(x, dims=[3]), 1 - side)
            else:
                r0, a0 = _one(x, None)
                r1, a1 = _one(torch.flip(x, dims=[3]), None)
            r = (r0.float() + r1.float()) / 2
            a = None if a0 is None else (a0.float() + a1.float()) / 2
        raws.append(r.cpu())
        ys.append(y)
        if a is not None:
            auxs.append(torch.sigmoid(a.view(a.shape[0], model.n_aux, 3)).cpu())
    raw = torch.cat(raws)
    ynp = torch.cat(ys).numpy()
    if want_aux:
        return raw, ynp, torch.cat(auxs).numpy()
    return raw, ynp


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument("--folds-csv", type=Path, default=PROJECT_ROOT / "reports" / "pipeline" / "folds.csv")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--backbone", choices=BACKBONES, default="resnet50")
    ap.add_argument("--head", choices=HEADS, default="cascade")
    ap.add_argument("--no-dual", action="store_true", help="global stream only")
    ap.add_argument("--local-frac", type=float, default=0.72)
    ap.add_argument("--sampler", choices=["natural", "sqrt", "stage6"], default="sqrt")
    ap.add_argument("--sord-alpha", type=float, default=1.5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone LR; head uses x10")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-train-batches", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-val-batches", type=int, default=None, help="smoke tests only")
    ap.add_argument("--aux-csv", type=Path, default=None, help="folds_features.csv; extra training targets only")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument(
        "--aux-jsn-med-weight",
        type=float,
        default=1.0,
        help="relative weight for aux_jsn_med in aux_loss (others=1; 2.0 = 2x medial JSN)",
    )
    ap.add_argument("--soft-kl12", type=float, default=0.0,
                    help="move eps mass between KL1 and KL2 in cascade/CORN targets (0=off)")
    ap.add_argument(
        "--soft-oa01",
        type=float,
        default=0.0,
        help="soften OA-gate targets for KL0/KL1 (KL0→eps, KL1→1-eps; 0=hard gate)",
    )
    ap.add_argument("--stream-mode", choices=STREAM_MODES, default="center")
    ap.add_argument("--medial-is-image-left-for-r", type=lambda s: s.lower() in ("1", "true", "yes"),
                    default=None, help="override medlat_side_mapping.json")
    ap.add_argument("--aug-interp", choices=["nearest", "bilinear"], default="nearest",
                    help="RandomAffine interpolation (nearest = backward compatible default)")
    ap.add_argument("--no-random-erasing", action="store_true",
                    help="disable RandomErasing in training transforms")
    ap.add_argument(
        "--fusion",
        choices=FUSION_MODES,
        default="concat",
        help="concat=baseline; scaled=learnable stream scales; gated=sigmoid gate on global only (local full)",
    )
    ap.add_argument(
        "--periphery-suppress",
        type=float,
        default=0.0,
        help="0..1; attenuate global-stream pixels outside --global-roi-frac box",
    )
    ap.add_argument(
        "--global-roi-frac",
        type=float,
        default=0.85,
        help="center ROI kept on the global stream when periphery-suppress > 0",
    )
    ap.add_argument(
        "--spatial-aux",
        action="store_true",
        help="predict jsn/ost med+lat from medial/lateral halves of the local feature map",
    )
    ap.add_argument(
        "--spatial-aux-global",
        action="store_true",
        help="also apply shared med/lat aux heads to global map halves (requires --spatial-aux)",
    )
    ap.add_argument(
        "--attn-band-weight",
        type=float,
        default=0.0,
        help="soft joint-band / frame-border attention prior weight (0=off)",
    )
    ap.add_argument(
        "--attn-border-frac",
        type=float,
        default=0.08,
        help="outer rim fraction used by the frame-border attention prior",
    )
    ap.add_argument(
        "--attn-global-only",
        action="store_true",
        help="apply attention prior only to the global stream (skip local band term)",
    )
    args = ap.parse_args(argv)
    if args.aux_csv is not None and not args.aux_csv.is_absolute():
        args.aux_csv = PROJECT_ROOT / args.aux_csv
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not args.no_amp

    fold_dir = args.out_dir / args.run_name / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir / "DONE").exists():
        assert_done_matches_args(fold_dir, args)
        print(f"[skip] {fold_dir} already DONE (config matches)")
        return

    use_aux = args.aux_csv is not None
    use_medlat = args.stream_mode == "medlat"
    use_spatial_aux = bool(args.spatial_aux)
    use_spatial_aux_global = bool(args.spatial_aux_global)
    if use_spatial_aux and not use_aux:
        raise SystemExit("--spatial-aux requires --aux-csv (jsn/ost med+lat targets)")
    if use_spatial_aux and use_medlat:
        raise SystemExit("--spatial-aux is for center dual only (not medlat)")
    if use_spatial_aux_global and not use_spatial_aux:
        raise SystemExit("--spatial-aux-global requires --spatial-aux")
    need_side = use_medlat or use_spatial_aux
    med_left_r = (
        load_medlat_mapping(args.medial_is_image_left_for_r)
        if need_side
        else True
    )
    rows = read_folds_csv(args.aux_csv if use_aux else args.folds_csv)
    if any(r["path"].split("/")[-1].lower().startswith("gp_") for r in rows):
        raise SystemExit("folds.csv contains G&P images; rebuild it with stage4_make_folds.py")
    rows = remap_rows_to_data_root(rows, args.data_root)
    train_rows = [r for r in rows if r["fold"] != args.fold]
    val_rows = [r for r in rows if r["fold"] == args.fold]
    # folds.csv is sorted by path (all KL0 first); mix classes so --max-val-batches subsets are usable
    val_rows = [val_rows[i] for i in np.random.default_rng(0).permutation(len(val_rows))]
    # side-aware: HFlip in the dataset with R/L swap; transforms omit RandomHorizontalFlip
    train_tf, eval_tf = build_transforms(
        args.img_size, horizontal_flip=not need_side,
        aug_interp=args.aug_interp, random_erasing=not args.no_random_erasing,
    )
    train_ds = FoldDataset(train_rows, args.data_root, train_tf, with_aux=use_aux,
                           with_side=need_side, flip_swap_side=need_side)
    val_ds = FoldDataset(val_rows, args.data_root, eval_tf, with_aux=use_aux,
                         with_side=need_side, flip_swap_side=False)

    w = sample_weights([r["label"] for r in train_rows], args.sampler)
    pin = dev.type == "cuda"
    if w is None:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                                  persistent_workers=args.num_workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=WeightedRandomSampler(w, len(train_ds), replacement=True),
                                  num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                                  persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = OrdinalNet(
        backbone=args.backbone, head=args.head, dual=not args.no_dual, local_frac=args.local_frac,
        dropout=args.dropout, pretrained=not args.no_pretrained, sord_alpha=args.sord_alpha,
        n_aux=len(AUX_COLS) if use_aux else 0, stream_mode=args.stream_mode,
        medial_is_image_left_for_r=med_left_r,
        fusion=args.fusion,
        periphery_suppress=args.periphery_suppress,
        global_roi_frac=args.global_roi_frac,
        spatial_aux=use_spatial_aux,
        spatial_aux_global=use_spatial_aux_global,
    ).to(dev)
    opt = torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": args.lr},
            {"params": list(model.head_parameters()), "lr": args.lr * 10},
        ],
        weight_decay=args.weight_decay,
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    steps_per_epoch = len(train_loader) if args.max_train_batches is None else min(len(train_loader), args.max_train_batches)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    state = {"epoch": 0, "step": 0, "best_qwk": -1.0, "best_epoch": 0, "bad_epochs": 0, "history": []}
    last = fold_dir / "last.pth"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema") is not None:
            ema.model.load_state_dict(ck["ema"])
        state = ck["state"]
        print(f"[resume] {fold_dir.name}: continuing after epoch {state['epoch']} "
              f"(best QWK {state['best_qwk']:.4f} @ epoch {state['best_epoch']})")
    if state.get("amp_disabled"):
        amp = False
        scaler = torch.amp.GradScaler("cuda", enabled=False)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"model": model.config(), "n_train": len(train_rows), "n_val": len(val_rows)})
    (fold_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[{args.run_name} fold {args.fold}] device={dev} amp={amp} train={len(train_rows)} "
          f"val={len(val_rows)} steps/epoch={steps_per_epoch}")

    eval_model = lambda: ema.model if ema is not None else model  # noqa: E731
    deadline = float(os.environ.get("PIPELINE_DEADLINE", "inf"))

    while state["epoch"] < args.epochs and state["bad_epochs"] < args.patience:
        epoch = state["epoch"] + 1
        model.train()
        t0 = time.time()
        run_loss, run_main, run_aux, run_attn, n_seen, n_skipped, streak = (
            0.0, 0.0, 0.0, 0.0, 0, 0, 0
        )
        use_attn = float(args.attn_band_weight) > 0.0
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches is not None and bi >= args.max_train_batches:
                break
            f = lr_factor(state["step"], warmup_steps, total_steps)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            x, y = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
            side = batch[-1].to(dev, non_blocking=True) if need_side else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                if use_aux or use_attn:
                    raw, raw_aux, acts = model.forward_train(x, side)
                else:
                    raw = model(x, side)
                    raw_aux, acts = None, None
            main_loss = model.loss(
                raw, y, soft_kl12=args.soft_kl12, soft_oa01=args.soft_oa01
            )
            loss = main_loss
            aux_loss = None
            attn_loss = None
            if use_aux:
                # AUX_COLS order: jsn_med, jsn_lat, ost_med, ost_lat
                feat_w = [float(args.aux_jsn_med_weight), 1.0, 1.0, 1.0]
                aux_loss = model.aux_loss(
                    raw_aux.float(),
                    batch[2].to(dev, non_blocking=True),
                    feat_weights=feat_w[: model.n_aux],
                )
                loss = loss + args.aux_weight * aux_loss
            if use_attn:
                include_local = model.dual and not bool(args.attn_global_only)
                attn_loss = attention_prior_loss(
                    acts, border_frac=args.attn_border_frac, include_local=include_local
                )
                loss = loss + args.attn_band_weight * attn_loss
            if not torch.isfinite(loss):
                n_skipped += 1
                streak += 1
                if amp and streak >= 3:
                    # some GPU/cuDNN combos (e.g. GTX 16xx + EfficientNet) produce NaN in fp16
                    print("  [warn] repeated non-finite fp16 losses -> switching to fp32 for this fold", flush=True)
                    amp = False
                    scaler = torch.amp.GradScaler("cuda", enabled=False)
                    state["amp_disabled"] = True
                state["step"] += 1
                continue
            streak = 0
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            state["step"] += 1
            run_loss += float(loss.detach()) * x.size(0)
            run_main += float(main_loss.detach()) * x.size(0)
            if aux_loss is not None:
                run_aux += float(aux_loss.detach()) * x.size(0)
            if attn_loss is not None:
                run_attn += float(attn_loss.detach()) * x.size(0)
            n_seen += x.size(0)

        em = eval_model()
        pout = predict(em, val_loader, dev, amp, args.max_val_batches)
        raw, yv = pout[0], pout[1]
        pred = em.decode(raw).numpy()
        m = metric_bundle(yv, pred)
        state["epoch"] = epoch
        score = m["qwk"] if np.isfinite(m["qwk"]) else -1.0
        improved = score > state["best_qwk"] + 1e-6 or not (fold_dir / "best.pth").exists()
        row = {
            "epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": run_loss / max(n_seen, 1),
            "val_qwk": m["qwk"], "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
            "val_kl0_to_kl1": m["kl0_to_kl1"], "val_kl2_to_kl1": m["kl2_to_kl1"],
            **{f"val_recall_kl{k}": m[f"recall_kl{k}"] for k in range(5)},
            "minutes": (time.time() - t0) / 60, "best": improved, "skipped_steps": n_skipped, "amp": amp,
        }
        if use_aux:
            row["train_main_loss"] = run_main / max(n_seen, 1)
            row["train_aux_loss"] = run_aux / max(n_seen, 1)
        if use_attn:
            row["train_attn_loss"] = run_attn / max(n_seen, 1)
        state["history"].append(row)
        if improved:
            state["best_qwk"], state["best_epoch"], state["bad_epochs"] = score, epoch, 0
            torch.save({"model_state_dict": em.state_dict(), "config": config, "epoch": epoch,
                        "val_qwk": m["qwk"]}, fold_dir / "best.pth")
        else:
            state["bad_epochs"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.model.state_dict() if ema is not None else None, "state": state},
                   fold_dir / "last.pth.tmp")
        (fold_dir / "last.pth.tmp").replace(last)
        with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(state["history"][0].keys()))
            wr.writeheader()
            wr.writerows(state["history"])
        tag = "*best*" if improved else f"(patience {state['bad_epochs']}/{args.patience})"
        aux_txt = f" main={row['train_main_loss']:.4f} aux={row['train_aux_loss']:.4f}" if use_aux else ""
        attn_txt = f" attn={row['train_attn_loss']:.4f}" if use_attn else ""
        print(f"  epoch {epoch:>2}/{args.epochs} loss={row['train_loss']:.4f}{aux_txt}{attn_txt} "
              f"val_qwk={m['qwk']:.4f} "
              f"f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} 0→1={m['kl0_to_kl1']} 2→1={m['kl2_to_kl1']} "
              f"{tag} [{row['minutes']:.1f} min]", flush=True)
        more = state["epoch"] < args.epochs and state["bad_epochs"] < args.patience
        if more and time.time() + 1.5 * row["minutes"] * 60 > deadline:
            print(f"[pause] session deadline near; stopped after epoch {epoch}. Re-run to resume.", flush=True)
            sys.exit(PAUSED_EXIT)

    best = torch.load(fold_dir / "best.pth", map_location=dev, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    pout = predict(model, val_loader, dev, amp, args.max_val_batches)
    raw, yv = pout[0], pout[1]
    aux_probs = pout[2] if len(pout) == 3 else None
    probs = model.to_probs(raw).numpy()
    pred = model.decode(raw).numpy()
    thr = model.threshold_probs(raw)
    n = len(yv)
    payload = dict(
        paths=np.array([r["path"] for r in val_rows[:n]]),
        patients=np.array([r["patient"] for r in val_rows[:n]]),
        labels=yv, raw=raw.numpy(), probs=probs, pred=pred,
        thr=thr.numpy() if thr is not None else np.zeros((n, 0)),
        fold=np.full(n, args.fold),
    )
    if aux_probs is not None:
        payload["aux_probs"] = aux_probs  # (N, n_aux, 3)
        payload["aux_names"] = np.array(AUX_COLS)
    np.savez_compressed(fold_dir / "oof.npz", **payload)
    final = metric_bundle(yv, pred)
    (fold_dir / "DONE").write_text(json.dumps({"best_epoch": state["best_epoch"], **{
        k: v for k, v in final.items() if k != "confusion_matrix"}}, indent=2), encoding="utf-8")
    last.unlink(missing_ok=True)
    print(f"[done] {args.run_name} fold {args.fold}: best epoch {state['best_epoch']} "
          f"val QWK {final['qwk']:.4f} F1 {final['macro_f1']:.4f} acc {final['accuracy']:.4f}")


if __name__ == "__main__":
    main()
