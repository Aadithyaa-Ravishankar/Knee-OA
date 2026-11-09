"""
aux_feature_eval.py
===================
Score aux-head OOF dumps against folds_features.csv, then try KL-from-features
on predicted grades (+ native KL probs) with fold-wise training.

Reads aux_oof.npz under <runs-dir>/<run-name>/fold*/ (from stage5_dump_aux.py).
Writes reports/pipeline/aux_feature_eval.md.

Example:
  python src/eval/aux_feature_eval.py --run-name r50_dual_cascade_sqrt_aux \\
      --runs-dir outputs/cv/runs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.protocol import metric_bundle
from src.training.stage4_cv_train import AUX_COLS

FEATURES_CSV = PROJECT_ROOT / "reports" / "pipeline" / "folds_features.csv"
OUT_MD = PROJECT_ROOT / "reports" / "pipeline" / "aux_feature_eval.md"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    ap.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    return ap.parse_args(argv)


def _parse_feat(cell: str) -> int | None:
    text = str(cell or "").strip()
    if text == "" or text.lower() == "nan":
        return None
    return int(float(text))


def load_feature_table(path: Path) -> dict[str, dict]:
    by_name: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            name = Path(r["path"]).name
            by_name[name] = {
                "path": r["path"],
                "label": int(r["label"]),
                "fold": int(r["fold"]),
                **{c: _parse_feat(r.get(c, "")) for c in AUX_COLS},
            }
    return by_name


def load_aux_oof(runs_dir: Path, run_name: str) -> list[dict]:
    root = runs_dir / run_name
    files = sorted(root.glob("fold*/aux_oof.npz"))
    if not files:
        raise SystemExit(f"no aux_oof.npz under {root}/fold*")
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        paths = d["paths"]
        labels = d["labels"]
        aux = d["aux_probs"]  # (N, n_aux, 3)
        probs = d["probs"]
        pred = d["pred"]
        fold = int(d["fold"][0]) if "fold" in d.files else int(f.parent.name.replace("fold", ""))
        names = list(d["aux_names"]) if "aux_names" in d.files else list(AUX_COLS[: aux.shape[1]])
        for i in range(len(labels)):
            rows.append({
                "path": str(paths[i]),
                "name": Path(str(paths[i])).name,
                "label": int(labels[i]),
                "fold": fold,
                "aux_probs": aux[i],
                "aux_names": names,
                "kl_probs": probs[i],
                "kl_pred": int(pred[i]),
                "feat_pred": (aux[i] > 0.5).sum(axis=-1).astype(int),
            })
    return rows


def feature_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=(0, 1, 2, 3)) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if len(y_true) else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }


def _cm_md(cm: list[list[int]], labels) -> str:
    header = "| true\\pred | " + " | ".join(str(l) for l in labels) + " |"
    sep = "|---|" + "|".join("---" for _ in labels) + "|"
    body = [f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]
    return "\n".join([header, sep, *body])


def kl_from_features_oof(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-fold-out logistic regression: predicted aux grades + KL probs -> KL."""
    folds = sorted({r["fold"] for r in rows})
    y_true, y_feat, y_native = [], [], []
    for hold in folds:
        train = [r for r in rows if r["fold"] != hold]
        test = [r for r in rows if r["fold"] == hold]
        if not train or not test:
            continue
        x_tr = np.stack([np.concatenate([r["feat_pred"].astype(float), r["kl_probs"]]) for r in train])
        y_tr = np.array([r["label"] for r in train], dtype=int)
        x_te = np.stack([np.concatenate([r["feat_pred"].astype(float), r["kl_probs"]]) for r in test])
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr)
        x_te = scaler.transform(x_te)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_te)
        y_true.extend(r["label"] for r in test)
        y_feat.extend(pred.tolist())
        y_native.extend(r["kl_pred"] for r in test)
    return np.asarray(y_true), np.asarray(y_feat), np.asarray(y_native)


def main(argv=None):
    args = parse_args(argv)
    feat_tab = load_feature_table(args.features_csv)
    rows = load_aux_oof(args.runs_dir, args.run_name)
    matched = [r for r in rows if r["name"] in feat_tab]
    if len(matched) != len(rows):
        print(f"warning: {len(rows) - len(matched)} dump rows not in features csv (by filename)")

    lines = [
        f"# Aux feature eval — `{args.run_name}`",
        "",
        f"Source: `{args.runs_dir / args.run_name}` (`aux_oof.npz`). "
        f"N dump images = {len(rows)}; matched to features = {len(matched)}.",
        "",
        "## Per-feature grades (cumulative P>0.5 count vs folds_features.csv)",
        "",
        "Non-missing feature rows only.",
        "",
    ]
    n_aux = rows[0]["aux_probs"].shape[0] if rows else 0
    names = list(rows[0]["aux_names"]) if rows else list(AUX_COLS[:n_aux])

    for j, col in enumerate(names):
        yt, yp = [], []
        for r in matched:
            gt = feat_tab[r["name"]].get(col)
            if gt is None:
                continue
            yt.append(gt)
            yp.append(int(r["feat_pred"][j]))
        m = feature_metrics(np.asarray(yt), np.asarray(yp))
        lines += [
            f"### `{col}`",
            "",
            f"- n={m['n']}  accuracy={m['accuracy']:.4f}  weighted κ={m['weighted_kappa']:.4f}",
            "",
            _cm_md(m["confusion_matrix"], [0, 1, 2, 3]),
            "",
        ]

    y_true, y_feat, y_native = kl_from_features_oof(matched)
    m_feat = metric_bundle(y_true, y_feat)
    m_nat = metric_bundle(y_true, y_native)
    lines += [
        "## KL-from-features (predicted aux grades + KL probs)",
        "",
        "Fold-wise: train LogisticRegression on other folds' OOF, predict held-out fold.",
        "Features = hard aux grades (4) concatenated with native KL softmax/cascade probs (5).",
        "",
        f"| decoder | n | QWK | accuracy | recall KL0 | KL1 | KL2 | KL3 | KL4 |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| KL-from-features | {len(y_true)} | {m_feat['qwk']:.4f} | {m_feat['accuracy']:.4f} | "
            f"{m_feat['recall_kl0']:.4f} | {m_feat['recall_kl1']:.4f} | {m_feat['recall_kl2']:.4f} | "
            f"{m_feat['recall_kl3']:.4f} | {m_feat['recall_kl4']:.4f} |"
        ),
        (
            f"| native KL head | {len(y_true)} | {m_nat['qwk']:.4f} | {m_nat['accuracy']:.4f} | "
            f"{m_nat['recall_kl0']:.4f} | {m_nat['recall_kl1']:.4f} | {m_nat['recall_kl2']:.4f} | "
            f"{m_nat['recall_kl3']:.4f} | {m_nat['recall_kl4']:.4f} |"
        ),
        "",
        "### Confusion — KL-from-features",
        "",
        _cm_md(m_feat["confusion_matrix"], [0, 1, 2, 3, 4]),
        "",
        "### Confusion — native KL head (same images)",
        "",
        _cm_md(m_nat["confusion_matrix"], [0, 1, 2, 3, 4]),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"KL-from-features QWK={m_feat['qwk']:.4f}  native QWK={m_nat['qwk']:.4f}")


if __name__ == "__main__":
    main()
