"""
stage8_gradcam.py
==================
Grad-CAM explainability for the Pipeline final champion (dual-stream OrdinalNet).

Uses per-stream Grad-CAM with local-stream un-zoom paste (src.evaluation.gradcam_dual).
Does not change training or locked-test scores.

Example:
  python src/training/stage8_gradcam.py --n 30 --split test --folds all
  python src/training/stage8_gradcam.py --n 8 --folds 0 --split test
  python src/training/stage8_gradcam.py --errors neighbor --n 0 --folds all --cam-folds 0
  python src/training/stage8_gradcam.py --source oof --errors neighbor --n 0 \\
      --run-name r50_dual_cascade_sqrt_aux_edgemask --runs-dir outputs/cv/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.decoding import apply_cutpoints, expected_grade, fit_cutpoints
from src.evaluation.gradcam_dual import (
    OrdinalNetAdapter,
    _norm01,
    band_mass,
    dual_gradcam,
    overlay_cam,
    tensor_to_gray_u8,
)
from src.models.ordinal_net import OrdinalNet
from src.preprocessing.dataset import IMG_EXTS, KL_GRADES, KneeOADataset
from src.training.stage4_cv_train import build_transforms
from src.training.stage4_oof_report import load_run

MEAN = [0.578707] * 3
STD = [0.145419] * 3


def _parse_folds(text: str) -> list[int] | None:
    text = text.strip().lower()
    if text in ("all", "*"):
        return None  # all available folds
    return [int(x) for x in text.split(",") if x.strip() != ""]


def _parse_target(text: str):
    text = text.strip().lower()
    if text in ("pred", "true"):
        return text
    return int(text)


def _error_kind(true_y: int, pred: int) -> str | None:
    if true_y == pred:
        return None
    d = abs(int(true_y) - int(pred))
    if d == 1:
        return "neighbor"
    return "far"


def load_fold_model(ckpt: Path, dev: torch.device) -> OrdinalNet:
    cfg_path = ckpt.parent / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    mc = cfg.get("model", cfg)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    # Prefer nested model config; fall back to top-level cfg for older checkpoints.
    mc = {**cfg, **mc}
    model = OrdinalNet.from_config(mc, dropout=float(cfg.get("dropout", 0.3)), pretrained=False).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model


def collect_split_samples(data_root: Path, split: str) -> list[tuple[Path, int]]:
    ds = KneeOADataset(data_root, split, transform=None)
    return [
        (fpath, label)
        for fpath, label in ds.samples
        if not fpath.name.lower().startswith("gp_") and fpath.suffix.lower() in IMG_EXTS
    ]


def stratified_pick(
    samples: list[tuple[Path, int]],
    n: int,
    seed: int = 0,
) -> list[tuple[Path, int]]:
    rng = np.random.default_rng(seed)
    by = {k: [s for s in samples if s[1] == k] for k in KL_GRADES}
    per = max(1, n // 5)
    out = []
    for k in KL_GRADES:
        pool = by[k]
        if not pool:
            continue
        take = min(per, len(pool))
        idxs = rng.choice(len(pool), size=take, replace=False)
        out.extend(pool[i] for i in idxs)
    # trim/pad toward n
    if len(out) > n:
        out = [out[i] for i in rng.choice(len(out), size=n, replace=False)]
    return out


@torch.no_grad()
def ensemble_predict(
    models: list[OrdinalNet],
    x: torch.Tensor,
    cuts: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    raws = [m(x).float() for m in models]
    raw_ens = torch.stack(raws).mean(0)
    probs = models[0].to_probs(raw_ens)[0].cpu().numpy()
    if cuts is not None:
        pred = int(apply_cutpoints(expected_grade(probs[None, :]), cuts)[0])
    else:
        pred = int(models[0].decode(raw_ens)[0].item())
    return pred, probs


def fit_run_cutpoints(run_dir: Path) -> np.ndarray:
    r = load_run(run_dir)
    if r is None:
        raise SystemExit(f"no oof.npz under {run_dir}; need OOF to fit cutpoints")
    return fit_cutpoints(expected_grade(r["probs"]), r["labels"])


def _cap_error_hits(hits: list, n: int, seed: int) -> list:
    """Cap hits stratified across (true, pred) pairs. n<=0 keeps all."""
    if n <= 0 or len(hits) <= n:
        return hits
    rng = np.random.default_rng(seed)
    by_pair: dict[tuple[int, int], list] = defaultdict(list)
    for h in hits:
        by_pair[(h[1], h[2])].append(h)
    pools = {k: list(v) for k, v in by_pair.items()}
    for k in pools:
        rng.shuffle(pools[k])
    out: list = []
    keys = sorted(pools.keys())
    while len(out) < n and any(pools[k] for k in keys):
        for k in keys:
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out


def select_error_samples(
    samples: list[tuple[Path, int]],
    models: list[OrdinalNet],
    eval_tf,
    dev: torch.device,
    mode: str,
    n: int,
    seed: int,
    cuts: np.ndarray | None = None,
) -> list[tuple[Path, int, int, np.ndarray]]:
    """Return (path, true, pred, probs) for misgrades matching mode (split scan)."""
    hits: list[tuple[Path, int, int, np.ndarray]] = []
    total = len(samples)
    for i, (fpath, true_y) in enumerate(samples, 1):
        img = Image.open(fpath).convert("L")
        x = eval_tf(img).unsqueeze(0).to(dev)
        pred, probs = ensemble_predict(models, x, cuts=cuts)
        kind = _error_kind(true_y, pred)
        if kind is None:
            pass
        elif mode == "neighbor" and kind != "neighbor":
            pass
        elif mode == "far" and kind != "far":
            pass
        else:
            hits.append((fpath, int(true_y), pred, probs))
        if i == 1 or i % 100 == 0 or i == total:
            print(f"  scanned {i}/{total}  errors_so_far={len(hits)}", flush=True)
    return _cap_error_hits(hits, n, seed)


def select_correct_samples(
    samples: list[tuple[Path, int]],
    models: list[OrdinalNet],
    eval_tf,
    dev: torch.device,
    n: int,
    seed: int,
    cuts: np.ndarray | None = None,
) -> list[tuple[Path, int, int, np.ndarray]]:
    """Stratified correct grades (true == pred), capped at n (0 = all)."""
    hits: list[tuple[Path, int, int, np.ndarray]] = []
    total = len(samples)
    for i, (fpath, true_y) in enumerate(samples, 1):
        img = Image.open(fpath).convert("L")
        x = eval_tf(img).unsqueeze(0).to(dev)
        pred, probs = ensemble_predict(models, x, cuts=cuts)
        if int(true_y) == int(pred):
            hits.append((fpath, int(true_y), pred, probs))
        if i == 1 or i % 100 == 0 or i == total:
            print(f"  scanned {i}/{total}  correct_so_far={len(hits)}", flush=True)
    if n <= 0 or len(hits) <= n:
        return hits
    # Stratify by true KL grade
    rng = np.random.default_rng(seed)
    by_kl: dict[int, list] = defaultdict(list)
    for h in hits:
        by_kl[h[1]].append(h)
    for k in by_kl:
        rng.shuffle(by_kl[k])
    out: list = []
    keys = sorted(by_kl.keys())
    while len(out) < n and any(by_kl[k] for k in keys):
        for k in keys:
            if by_kl[k] and len(out) < n:
                out.append(by_kl[k].pop())
    return out


def write_correct_proof_report(out_dir: Path, rows: list[dict], meta: dict) -> None:
    by_kl: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_kl[int(r["true"])].append(r)
    lines = [
        "# Grad-CAM proof — correctly graded images",
        "",
        f"- run: `{meta.get('run_name')}`  **split: {meta.get('split')}**",
        f"- decode: `{meta.get('decode')}`",
        f"- n correct panels: **{len(rows)}** (stratified across KL when capped)",
        f"- pred: 5-fold ensemble; CAM: {meta.get('cam_folds')}",
        f"- mean combined / global / local band: "
        f"**{meta.get('mean_band_combined'):.3f} / {meta.get('mean_band_global'):.3f} / "
        f"{meta.get('mean_band_local'):.3f}** "
        f"(uniform baseline {meta.get('band_mass_uniform_baseline'):.3f})",
        "",
        "High band mass on **correct** grades supports that the model attends to the joint "
        "region when it grades right — not only on error cases.",
        "",
        "## By true KL",
        "",
        "| KL | n | band_combined | band_local | mean P(true) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in sorted(by_kl.keys()):
        grp = by_kl[k]
        bc = float(np.mean([g["band_mass_combined"] for g in grp]))
        bl = float(np.mean([g["band_mass_local"] for g in grp if g.get("band_mass_local") is not None]))
        pt = float(np.mean([g["probs"][g["true"]] for g in grp]))
        lines.append(f"| {k} | {len(grp)} | {bc:.3f} | {bl:.3f} | {pt:.3f} |")
    lines += [
        "",
        "## How to read",
        "",
        "- Panels: `panels/KL{k}/*_panel.png` = original | combined | global | local.",
        "- Target = predicted (= true) class.",
        "- Band mass >> uniform baseline → attention concentrated on the joint band.",
        "",
    ]
    (out_dir / "PROOF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "proof_summary.json").write_text(
        json.dumps({"meta": meta, "n": len(rows), "by_kl": {str(k): len(v) for k, v in by_kl.items()}}, indent=2),
        encoding="utf-8",
    )


def collect_oof_error_samples(
    run_dir: Path,
    data_root: Path,
    mode: str,
    n: int,
    seed: int,
) -> list[tuple[Path, int, int, np.ndarray, int]]:
    """Neighbor/far/all misgrades from fold OOF preds (fold-val images only).

    Returns (path, true, pred, probs, fold_id). Predictions come from each fold's
    oof.npz — same QWK numbers, no test images.
    """
    hits: list[tuple[Path, int, int, np.ndarray, int]] = []
    oof_files = sorted(run_dir.glob("fold*/oof.npz"))
    if not oof_files:
        raise SystemExit(f"no oof.npz under {run_dir}; need finished folds")
    for oof_path in oof_files:
        fold_id = int(oof_path.parent.name.replace("fold", ""))
        z = np.load(oof_path, allow_pickle=True)
        paths = z["paths"]
        labels = z["labels"].astype(int)
        preds = z["pred"].astype(int)
        probs = z["probs"].astype(np.float32)
        for i in range(len(labels)):
            true_y, pred = int(labels[i]), int(preds[i])
            kind = _error_kind(true_y, pred)
            if kind is None:
                continue
            if mode == "neighbor" and kind != "neighbor":
                continue
            if mode == "far" and kind != "far":
                continue
            rel = str(paths[i])
            fpath = Path(rel)
            if not fpath.is_file():
                fpath = data_root / rel
            if not fpath.is_file():
                print(f"  skip missing {rel}", flush=True)
                continue
            hits.append((fpath, true_y, pred, probs[i], fold_id))
        print(f"  fold{fold_id}: oof n={len(labels)} errors_so_far={len(hits)}", flush=True)
    return _cap_error_hits(hits, n, seed)


def write_error_report(out_dir: Path, rows: list[dict], meta: dict) -> None:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[f"{r['true']}->{r['pred']}"].append(r)

    pair_stats = []
    for key in sorted(by_pair.keys(), key=lambda s: (int(s.split("->")[0]), int(s.split("->")[1]))):
        grp = by_pair[key]
        pair_stats.append(
            {
                "pair": key,
                "n": len(grp),
                "mean_band_mass_combined": float(np.mean([g["band_mass_combined"] for g in grp])),
                "mean_band_mass_local": float(
                    np.mean([g["band_mass_local"] for g in grp if g["band_mass_local"] is not None])
                ),
                "mean_pred_prob": float(np.mean([g["probs"][g["pred"]] for g in grp])),
                "mean_true_prob": float(np.mean([g["probs"][g["true"]] for g in grp])),
                "stems": [g["stem"] for g in grp],
            }
        )

    report = {
        **meta,
        "n_errors": len(rows),
        "pair_stats": pair_stats,
        "mean_band_mass_combined": float(np.mean([r["band_mass_combined"] for r in rows])) if rows else None,
        "band_mass_uniform_baseline": band_mass(np.ones((224, 224), dtype=np.float32)),
    }
    (out_dir / "error_pairs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Grad-CAM neighbor / misgrade analysis",
        "",
        f"- run: `{meta.get('run_name')}`  split: `{meta.get('split')}`",
        f"- errors mode: `{meta.get('errors')}`  n: {len(rows)}",
        f"- pred folds: {meta.get('folds')}  cam folds: {meta.get('cam_folds')}",
        f"- mean combined band mass: {report['mean_band_mass_combined']:.3f} "
        f"(uniform baseline {report['band_mass_uniform_baseline']:.3f})",
        "",
        "## By confusion pair (true → pred)",
        "",
        "| pair | n | band_combined | band_local | mean P(pred) | mean P(true) |",
        "|------|---|---------------|------------|--------------|--------------|",
    ]
    for p in pair_stats:
        lines.append(
            f"| {p['pair']} | {p['n']} | {p['mean_band_mass_combined']:.3f} | "
            f"{p['mean_band_mass_local']:.3f} | {p['mean_pred_prob']:.3f} | {p['mean_true_prob']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- Panels: `*_panel.png` = original | combined | global | local.",
            "- Target = **predicted** class (why the model chose the wrong grade).",
            "- Low band mass → attention off the joint line (possible shortcut / poor localization).",
            "- High band mass but still wrong → joint-focused but ordinal boundary confusion "
            "(typical for KL0↔1 and KL1↔2).",
            "",
        ]
    )
    (out_dir / "ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def make_panel(
    gray: np.ndarray,
    combined: np.ndarray,
    streams: list[np.ndarray],
    title: str,
) -> np.ndarray:
    tiles = [
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        overlay_cam(gray, combined),
    ]
    for s in streams:
        tiles.append(overlay_cam(gray, s))
    labels = ["original", "combined", "global"] + [f"local{i}" for i in range(max(0, len(streams) - 1))]
    row = np.concatenate(tiles, axis=1)
    header = np.zeros((28, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title[:120], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for j, name in enumerate(labels[: len(tiles)]):
        x0 = j * gray.shape[1] + 4
        cv2.putText(row, name, (x0, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.concatenate([header, row], axis=0)


def resolve_target(mode, true_y: int, pred: int) -> int | None:
    if mode == "pred":
        return None  # dual_gradcam uses pred
    if mode == "true":
        return int(true_y)
    return int(mode)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--run-name", default="r50_dual_cascade_sqrt_aux")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Dataset_combined")
    ap.add_argument(
        "--source",
        choices=("split", "oof"),
        default="split",
        help="split=train/val/test folder; oof=fold-validation images + OOF preds (keeps test locked)",
    )
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--folds", default="all", help="'all' or comma list e.g. 0,1 (ensemble pred)")
    ap.add_argument(
        "--cam-folds",
        default=None,
        help="folds used for Grad-CAM only (default: same as --folds). Ignored for --source oof "
        "(each image uses its own fold model).",
    )
    ap.add_argument("--target", default="pred", help="pred | true | KL int 0..4")
    ap.add_argument("--n", type=int, default=30, help="max images; 0 = all (with --errors)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--errors",
        choices=("none", "neighbor", "far", "all", "correct"),
        default="none",
        help="none=stratified sample; neighbor/far/all=misgrades; correct=true==pred proof set",
    )
    ap.add_argument(
        "--decode",
        choices=("native", "expected_cutpoints"),
        default="native",
        help="native cascade decode, or OOF-fit expected-grade cutpoints (final recipe)",
    )
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "pipeline" / "gradcam")
    ap.add_argument("--paths", nargs="*", default=None, help="optional explicit image paths")
    args = ap.parse_args(argv)
    if args.source == "oof" and args.errors == "none":
        raise SystemExit("--source oof requires --errors neighbor|far|all|correct")
    if args.source == "oof" and args.errors == "correct":
        raise SystemExit("--errors correct is for --source split (use test); OOF correct not wired yet")

    run_dir = args.runs_dir / args.run_name
    all_ckpts = sorted(run_dir.glob("fold*/best.pth"))
    if not all_ckpts:
        raise SystemExit(f"no best.pth under {run_dir}")

    cuts = None
    if args.decode == "expected_cutpoints":
        cuts = fit_run_cutpoints(run_dir)
        print(f"OOF-fit cutpoints: {np.asarray(cuts).round(4).tolist()}", flush=True)

    def _filter_ckpts(text: str | None) -> list[Path]:
        fold_ids = _parse_folds(text) if text is not None else None
        if fold_ids is None:
            return list(all_ckpts)
        want = set(fold_ids)
        out = [c for c in all_ckpts if int(c.parent.name.replace("fold", "")) in want]
        if not out:
            raise SystemExit(f"no checkpoints for folds {fold_ids}")
        return out

    ckpts_pred = _filter_ckpts(args.folds)
    ckpts_cam = _filter_ckpts(args.cam_folds if args.cam_folds is not None else args.folds)
    target_mode = _parse_target(args.target)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Share loaded weights when pred/cam fold sets overlap
    loaded: dict[Path, OrdinalNet] = {}
    for c in sorted(set(ckpts_pred) | set(ckpts_cam), key=str):
        loaded[c] = load_fold_model(c, dev)
    models_pred = [loaded[c] for c in ckpts_pred]
    models_cam = [loaded[c] for c in ckpts_cam]

    img_size = int(json.loads((ckpts_pred[0].parent / "config.json").read_text(encoding="utf-8")).get("img_size", 224))
    _, eval_tf = build_transforms(img_size, horizontal_flip=False)

    precomputed: dict[str, tuple[int, np.ndarray]] = {}
    oof_fold_of: dict[str, int] = {}
    if args.paths:
        samples = []
        for p in args.paths:
            path = Path(p)
            if not path.is_file():
                path = args.data_root / p
            try:
                label = int(path.parent.name)
            except ValueError:
                label = -1
            samples.append((path, label))
    elif args.source == "oof":
        print(
            f"loading OOF fold-val {args.errors} errors from {run_dir} (test untouched)...",
            flush=True,
        )
        err_rows = collect_oof_error_samples(
            run_dir, args.data_root, mode=args.errors, n=args.n, seed=args.seed
        )
        print(f"selected {len(err_rows)} OOF error images for Grad-CAM", flush=True)
        samples = [(p, y) for p, y, _, _, _ in err_rows]
        for p, _y, pred, probs, fold_id in err_rows:
            precomputed[str(p)] = (pred, probs)
            oof_fold_of[str(p)] = int(fold_id)
        # ensure every OOF fold model is loaded for matched CAM
        for fold_id in sorted(set(oof_fold_of.values())):
            ck = run_dir / f"fold{fold_id}" / "best.pth"
            if ck not in loaded:
                loaded[ck] = load_fold_model(ck, dev)
    elif args.errors == "correct":
        print(
            f"scanning {args.split} for correctly graded images "
            f"(decode={args.decode}, ensemble {len(models_pred)} folds)...",
            flush=True,
        )
        err_rows = select_correct_samples(
            collect_split_samples(args.data_root, args.split),
            models_pred,
            eval_tf,
            dev,
            n=args.n,
            seed=args.seed,
            cuts=cuts,
        )
        print(
            f"selected {len(err_rows)} correct images for Grad-CAM (cam folds={len(models_cam)})",
            flush=True,
        )
        samples = [(p, y) for p, y, _, _ in err_rows]
        for p, _y, pred, probs in err_rows:
            precomputed[str(p)] = (pred, probs)
    elif args.errors != "none":
        print(
            f"scanning {args.split} for {args.errors} errors (ensemble {len(models_pred)} folds)...",
            flush=True,
        )
        err_rows = select_error_samples(
            collect_split_samples(args.data_root, args.split),
            models_pred,
            eval_tf,
            dev,
            mode=args.errors,
            n=args.n,
            seed=args.seed,
            cuts=cuts,
        )
        print(
            f"selected {len(err_rows)} error images for Grad-CAM (cam folds={len(models_cam)})",
            flush=True,
        )
        samples = [(p, y) for p, y, _, _ in err_rows]
        for p, _y, pred, probs in err_rows:
            precomputed[str(p)] = (pred, probs)
    else:
        samples = stratified_pick(collect_split_samples(args.data_root, args.split), args.n, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    # organize panels by confusion pair / KL when doing error or correct analysis
    if args.errors != "none":
        for sub in ("panels",):
            (args.out / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for fpath, true_y in samples:
        img = Image.open(fpath).convert("L")
        x = eval_tf(img).unsqueeze(0).to(dev)

        if str(fpath) in precomputed:
            pred_ens, probs_ens = precomputed[str(fpath)]
        else:
            pred_ens, probs_ens = ensemble_predict(models_pred, x, cuts=cuts)
        tgt = resolve_target(target_mode, true_y, pred_ens)

        if str(fpath) in oof_fold_of:
            fold_id = oof_fold_of[str(fpath)]
            cam_models = [loaded[run_dir / f"fold{fold_id}" / "best.pth"]]
            cam_tag = f"oof_fold{fold_id}"
        else:
            cam_models = models_cam
            cam_tag = f"cam_folds={len(models_cam)}"

        cams_c, cams_streams = [], []
        for m in cam_models:
            adapter = OrdinalNetAdapter(m)
            out = dual_gradcam(adapter, x, target_class=tgt)
            cams_c.append(out["combined"])
            cams_streams.append(out["streams"])

        combined = _norm01(np.mean(np.stack(cams_c, axis=0), axis=0))
        n_streams = len(cams_streams[0])
        streams = [
            _norm01(np.mean(np.stack([s[i] for s in cams_streams], axis=0), axis=0))
            for i in range(n_streams)
        ]

        gray = tensor_to_gray_u8(x, MEAN, STD)
        err = _error_kind(true_y, pred_ens)
        title = (
            f"{fpath.name}  true={true_y}  pred={pred_ens}  "
            f"target={pred_ens if tgt is None else tgt}  "
            f"{err or 'exact'}  decode={args.decode}  {cam_tag}"
        )
        panel = make_panel(gray, combined, streams, title)
        if args.errors == "correct":
            pair_dir = args.out / "panels" / f"KL{true_y}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            out_png = pair_dir / f"{fpath.stem}_panel.png"
        elif args.errors != "none":
            pair_dir = args.out / "panels" / f"{true_y}_to_{pred_ens}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            out_png = pair_dir / f"{fpath.stem}_panel.png"
        else:
            out_png = args.out / f"{fpath.stem}_panel.png"
        cv2.imwrite(str(out_png), panel)

        row = {
            "path": str(fpath),
            "stem": fpath.stem,
            "true": int(true_y),
            "pred": pred_ens,
            "error_kind": err,
            "target": int(pred_ens if tgt is None else tgt),
            "probs": [float(p) for p in probs_ens],
            "band_mass_combined": band_mass(combined),
            "band_mass_global": band_mass(streams[0]) if streams else None,
            "band_mass_local": band_mass(streams[1]) if len(streams) > 1 else None,
            "panel": str(out_png.resolve().relative_to(PROJECT_ROOT.resolve())),
            "source": args.source,
            "oof_fold": oof_fold_of.get(str(fpath)),
            "n_pred_folds": len(models_pred),
            "n_cam_folds": len(cam_models),
        }
        rows.append(row)
        print(
            f"[{len(rows)}/{len(samples)}] {fpath.name}: true={true_y} pred={pred_ens} ({err or 'exact'}) "
            f"band={row['band_mass_combined']:.3f} -> {out_png.relative_to(args.out)}",
            flush=True,
        )

    mean_g = (
        float(np.mean([r["band_mass_global"] for r in rows if r.get("band_mass_global") is not None]))
        if any(r.get("band_mass_global") is not None for r in rows)
        else None
    )
    mean_l = (
        float(np.mean([r["band_mass_local"] for r in rows if r.get("band_mass_local") is not None]))
        if any(r.get("band_mass_local") is not None for r in rows)
        else None
    )
    mean_c = float(np.mean([r["band_mass_combined"] for r in rows])) if rows else None
    summary = {
        "run_name": args.run_name,
        "source": args.source,
        "split": args.split if args.source == "split" else "oof_fold_val",
        "errors": args.errors,
        "decode": args.decode,
        "oof_cutpoints": None if cuts is None else np.asarray(cuts, dtype=float).tolist(),
        "n_images": len(rows),
        "n_pred_folds": len(models_pred),
        "n_cam_folds": len(models_cam),
        "folds": [int(c.parent.name.replace("fold", "")) for c in ckpts_pred],
        "cam_folds": (
            "per-image oof fold"
            if args.source == "oof"
            else [int(c.parent.name.replace("fold", "")) for c in ckpts_cam]
        ),
        "target_mode": args.target,
        "mean_band_mass_combined": mean_c,
        "mean_band_mass_global": mean_g,
        "mean_band_mass_local": mean_l,
        "band_mass_uniform_baseline": band_mass(np.ones((224, 224), dtype=np.float32)),
        "rows": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if args.errors == "correct":
        write_correct_proof_report(
            args.out,
            rows,
            {
                "run_name": args.run_name,
                "split": summary["split"],
                "decode": args.decode,
                "cam_folds": summary["cam_folds"],
                "mean_band_combined": mean_c,
                "mean_band_global": mean_g,
                "mean_band_local": mean_l,
                "band_mass_uniform_baseline": summary["band_mass_uniform_baseline"],
            },
        )
    elif args.errors != "none":
        write_error_report(
            args.out,
            rows,
            {
                "run_name": args.run_name,
                "split": summary["split"],
                "source": args.source,
                "errors": args.errors,
                "folds": summary["folds"],
                "cam_folds": summary["cam_folds"],
            },
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
