"""
stage7_decode_calibrate.py
===========================
Inference-only decode calibration on OOF (fold validation; test untouched).

For a cascade run with oof.npz (raw = [bin, oa0, oa1, oa2]):
  1) native decode (saved pred / OA gate 0.5)
  2) cross-fit OA-threshold sweep (fit thr on other folds, apply to held-out fold)
  3) cross-fit expected-grade cutpoints (existing decoding.py helper)

Example:
  python src/training/stage7_decode_calibrate.py \\
      --run-name r50_dual_cascade_sqrt_aux_attn_spatial \\
      --runs-dir outputs/cv/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.decoding import crossfit_cutpoints, expected_grade
from src.evaluation.ensemble import s6_predict_with_threshold
from src.evaluation.protocol import metric_bundle
from src.training.stage4_oof_report import load_run


def _qwk(y, p) -> float:
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def decode_oa(raw: np.ndarray, thr: float) -> np.ndarray:
    t = torch.from_numpy(np.asarray(raw, dtype=np.float32))
    return s6_predict_with_threshold(t[:, :1], t[:, 1:], oa_threshold=float(thr)).numpy()


def crossfit_oa_threshold(
    raw: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """For each fold, pick OA thr maximizing QWK on other folds; apply to this fold."""
    pred = np.zeros(len(y), dtype=np.int64)
    per_fold = {}
    for g in np.unique(folds):
        m = folds == g
        best_thr, best_q = 0.5, -1.0
        for thr in grid:
            q = _qwk(y[~m], decode_oa(raw[~m], thr))
            if q > best_q + 1e-12:
                best_q, best_thr = q, float(thr)
        pred[m] = decode_oa(raw[m], best_thr)
        per_fold[int(g)] = {"oa_threshold": best_thr, "fit_qwk": best_q}
    return pred, per_fold


def fit_global_oa_threshold(raw: np.ndarray, y: np.ndarray, grid: np.ndarray) -> dict:
    """In-sample sweep (for reporting only — not used as the fair OOF score)."""
    rows = []
    best = {"oa_threshold": 0.5, "qwk": -1.0}
    for thr in grid:
        p = decode_oa(raw, thr)
        q = _qwk(y, p)
        rows.append({"oa_threshold": float(thr), "qwk": q})
        if q > best["qwk"] + 1e-12:
            best = {"oa_threshold": float(thr), "qwk": q, "pred": p}
    return {"best": {k: best[k] for k in ("oa_threshold", "qwk")}, "curve": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--thr-lo", type=float, default=0.30)
    ap.add_argument("--thr-hi", type=float, default=0.70)
    ap.add_argument("--thr-step", type=float, default=0.02)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report dir (default: reports/pipeline/decode_calibrate_<run>)",
    )
    args = ap.parse_args(argv)

    run_dir = args.runs_dir / args.run_name
    r = load_run(run_dir)
    if r is None:
        raise SystemExit(f"no oof.npz under {run_dir}")
    if "raw" not in r:
        # load_run drops raw; reload with raw
        parts = sorted(run_dir.glob("fold*/oof.npz"))
        data = [np.load(p, allow_pickle=False) for p in parts]
        if "raw" not in data[0].files:
            raise SystemExit("oof.npz missing raw logits; cannot sweep OA threshold")
        raw = np.concatenate([d["raw"] for d in data])
        order = np.argsort(np.concatenate([d["paths"] for d in data]))
        raw = raw[order]
    else:
        raw = r["raw"]

    # load_run currently omits raw — always reload it aligned to sorted paths
    parts = sorted(run_dir.glob("fold*/oof.npz"))
    data = [np.load(p, allow_pickle=False) for p in parts]
    paths = np.concatenate([d["paths"] for d in data])
    order = np.argsort(paths)
    raw = np.concatenate([d["raw"] for d in data])[order]
    y = r["labels"]
    folds = r["fold"]
    native = r["pred"]
    probs = r["probs"]

    grid = np.round(np.arange(args.thr_lo, args.thr_hi + 1e-9, args.thr_step), 4)
    oa_pred, oa_folds = crossfit_oa_threshold(raw, y, folds, grid)
    cut_pred = crossfit_cutpoints(expected_grade(probs), y, folds)
    insample = fit_global_oa_threshold(raw, y, grid)

    variants = {
        "native_oa0.50": native,
        "crossfit_oa_threshold": oa_pred,
        "crossfit_expected_cutpoints": cut_pred,
    }
    results = {name: metric_bundle(y, p) for name, p in variants.items()}

    out = args.out or (PROJECT_ROOT / "reports" / "pipeline" / f"decode_calibrate_{args.run_name}")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "n": int(len(y)),
        "n_folds": int(r["n_folds_done"]),
        "thr_grid": grid.tolist(),
        "crossfit_oa_per_fold": oa_folds,
        "insample_oa_best": insample["best"],
        "insample_oa_curve": insample["curve"],
        "results": results,
    }
    (out / "decode_calibrate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Decode calibration — `{args.run_name}` (OOF only)",
        "",
        f"n={len(y)}, folds={r['n_folds_done']}. Test untouched.",
        "",
        "| decode | QWK | Macro-F1 | Acc | KL0→1 | KL2→1 | Recall KL0..4 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["qwk"]):
        rec = "/".join(f"{m[f'recall_kl{k}']:.2f}" for k in range(5))
        lines.append(
            f"| {name} | {m['qwk']:.4f} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | "
            f"{m['kl0_to_kl1']} | {m['kl2_to_kl1']} | {rec} |"
        )
    mean_thr = float(np.mean([v["oa_threshold"] for v in oa_folds.values()]))
    lines += [
        "",
        f"- Cross-fit OA thresholds by fold: "
        + ", ".join(f"f{k}={v['oa_threshold']:.2f}" for k, v in sorted(oa_folds.items())),
        f"- Mean cross-fit OA threshold ≈ **{mean_thr:.2f}** "
        f"(in-sample best thr={insample['best']['oa_threshold']:.2f}, "
        f"optimistic QWK={insample['best']['qwk']:.4f})",
        "",
        "Fair scores are the **crossfit_*** rows. Use those for selection; "
        "apply the mean OA threshold (or re-fit cutpoints on all OOF) only at locked test if OOF improves.",
    ]
    (out / "decode_calibrate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out / 'decode_calibrate.md'}")


if __name__ == "__main__":
    main()
