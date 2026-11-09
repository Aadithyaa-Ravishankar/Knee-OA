"""
stage4_oof_report.py
=====================
Out-of-fold (OOF) comparison of Pipeline runs. This is the model-selection step:
it uses only fold-validation predictions, never the test split.

For each run directory with oof.npz files it reports:
  - OOF metrics + bootstrap 95% CI for the native decode
  - expected grade + cutpoints, cross-fit by fold (cutpoints for fold k fit on the other folds)
  - paired bootstrap difference vs --baseline (same images)

Usage:
  python src/training/stage4_oof_report.py [--runs-dir outputs/cv/runs] [--baseline r50_dual_cascade_sqrt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.decoding import crossfit_cutpoints, expected_grade
from src.evaluation.protocol import bootstrap_ci, metric_bundle, paired_bootstrap_diff


def load_run(run_dir: Path) -> dict | None:
    parts = sorted(run_dir.glob("fold*/oof.npz"))
    if not parts:
        return None
    data = [np.load(p, allow_pickle=False) for p in parts]
    keys = ["paths", "labels", "probs", "pred", "fold"]
    if all("patients" in d.files for d in data):
        keys.append("patients")
    out = {k: np.concatenate([d[k] for d in data]) for k in keys}
    order = np.argsort(out["paths"])
    out = {k: v[order] for k, v in out.items()}
    out["n_folds_done"] = len(parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--baseline", default=None, help="run name to compare against")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "pipeline" / "oof_report")
    args = ap.parse_args()

    runs = {}
    for d in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        r = load_run(d)
        if r is not None:
            runs[d.name] = r
    if not runs:
        sys.exit(f"no oof.npz found under {args.runs_dir}")

    results, preds = {}, {}
    for name, r in runs.items():
        y = r["labels"]
        native = r["pred"]
        patients = r.get("patients")
        cut = crossfit_cutpoints(expected_grade(r["probs"]), y, r["fold"]) if r["n_folds_done"] > 1 else native
        for tag, p in (("native", native), ("cutpoints", cut)):
            key = f"{name} [{tag}]"
            preds[key] = (r["paths"], p, y, patients)
            results[key] = {
                "n_folds": r["n_folds_done"], "n": int(len(y)),
                "metrics": metric_bundle(y, p),
                "ci": bootstrap_ci(y, p, patients=patients),
                "bootstrap": "patient" if patients is not None else "image",
            }

    paired = {}
    if args.baseline:
        base_key = f"{args.baseline} [native]"
        if base_key not in preds:
            sys.exit(f"baseline {args.baseline} not found; runs: {list(runs)}")
        bp, bpred, by, bpat = preds[base_key]
        for key, (paths, p, y, patients) in preds.items():
            if key == base_key:
                continue
            common, ib, ik = np.intersect1d(bp, paths, return_indices=True)
            if len(common) < 50:
                continue
            pat = None if bpat is None or patients is None else bpat[ib]
            paired[key] = {
                "n_common": int(len(common)),
                "bootstrap": "patient" if pat is not None else "image",
                **paired_bootstrap_diff(by[ib], bpred[ib], p[ik], patients=pat),
            }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oof_report.json").write_text(json.dumps({"results": results, "paired_vs_baseline": paired},
                                                         indent=2), encoding="utf-8")
    lines = ["# Pipeline OOF report (fold-validation only; test untouched)", "",
             "| Run [decode] | folds | n | QWK [95% CI] | Macro-F1 | Acc | Recall KL0..KL4 | KL0→1 | KL2→1 | ΔQWK vs baseline [CI] |",
             "|---|---:|---:|---|---:|---:|---|---:|---:|---|"]
    for key, r in sorted(results.items(), key=lambda kv: -kv[1]["metrics"]["qwk"]):
        m, ci = r["metrics"], r["ci"]["qwk"]
        rec = "/".join(f"{m[f'recall_kl{k}']:.2f}" for k in range(5))
        d = paired.get(key, {}).get("qwk")
        dtxt = f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{' *' if d['significant'] else ''}" if d else "—"
        lines.append(f"| {key} | {r['n_folds']} | {r['n']} | {m['qwk']:.4f} [{ci['lo']:.3f}, {ci['hi']:.3f}] | "
                     f"{m['macro_f1']:.4f} | {m['accuracy']:.4f} | {rec} | {m['kl0_to_kl1']} | {m['kl2_to_kl1']} | {dtxt} |")
    lines += [
        "",
        "`*` = 95% CI of the difference excludes 0. The Δ column is computed only on images both runs "
        "predicted (same folds), so it is the fair comparison when runs have different numbers of folds done; "
        "the QWK column is not. QWK CIs and paired Δ use **patient-clustered** bootstrap when `patients` "
        "is present in oof.npz (resampling patients, keeping all of their images).",
    ]
    (args.out / "oof_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
