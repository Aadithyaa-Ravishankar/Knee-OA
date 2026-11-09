"""
protocol.py
===========
Pipeline evaluation protocol.

- Patient-grouped, label-stratified K folds over Chen train+val (G&P excluded).
- Bootstrap confidence intervals and paired bootstrap differences.
- Test split stays locked; nothing here reads it.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

KL_LABELS = [0, 1, 2, 3, 4]
IMG_EXTS = {".png", ".jpg", ".jpeg"}
PATIENT_RE = re.compile(r"^(\d+)[LR]$")


def patient_id(stem: str) -> str:
    m = PATIENT_RE.match(stem)
    return m.group(1) if m else stem


def collect_chen_rows(data_root: Path, splits=("train", "val")) -> list[dict]:
    """All non-G&P images in the given splits, with relative path, label, patient."""
    rows = []
    for split in splits:
        for grade in KL_LABELS:
            d = data_root / split / str(grade)
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in IMG_EXTS or p.name.lower().startswith("gp_"):
                    continue
                rows.append(
                    {
                        "path": f"{split}/{grade}/{p.name}",
                        "label": grade,
                        "patient": patient_id(p.stem),
                        "source_split": split,
                    }
                )
    return rows


def make_folds(rows: list[dict], n_folds: int = 5, seed: int = 42) -> list[dict]:
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["patient"] for r in rows])
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = [dict(r) for r in rows]
    for fold, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        for i in val_idx:
            out[i]["fold"] = fold
    return out


def write_folds_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "patient", "source_split", "fold"])
        w.writeheader()
        w.writerows(rows)


def read_folds_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["fold"] = int(r["fold"])
    return rows


# ── metrics ──────────────────────────────────────────────────────────────────

def metric_bundle(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=KL_LABELS)
    out = {
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, labels=KL_LABELS, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "kl0_to_kl1": int(cm[0, 1]),
        "kl1_to_kl2": int(cm[1, 2]),
        "kl2_to_kl1": int(cm[2, 1]),
        "pred_kl1_count": int(cm[:, 1].sum()),
        "true_kl1_count": int(cm[1].sum()),
    }
    for k in KL_LABELS:
        n = cm[k].sum()
        out[f"recall_kl{k}"] = float(cm[k, k] / n) if n else 0.0
    out["confusion_matrix"] = cm.tolist()
    return out


_CI_KEYS = ["qwk", "macro_f1", "accuracy"] + [f"recall_kl{k}" for k in KL_LABELS]


def _fast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    n = cm.sum()
    w = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16.0
    expected = np.outer(cm.sum(1), cm.sum(0)) / max(n, 1)
    denom = (w * expected).sum()
    qwk = 1.0 - (w * cm).sum() / denom if denom > 0 else 0.0
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(5), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(5), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(5), where=(prec + rec) > 0)
    out = {"qwk": qwk, "macro_f1": f1.mean(), "accuracy": tp.sum() / max(n, 1)}
    for k in KL_LABELS:
        out[f"recall_kl{k}"] = rec[k]
    return out


def _patient_bootstrap_indices(
    rng: np.random.Generator, patients: np.ndarray, n_boot: int
) -> list[np.ndarray]:
    """Resample patients with replacement; each draw keeps all images of chosen patients."""
    patients = np.asarray(patients)
    uniq, inverse = np.unique(patients, return_inverse=True)
    buckets = [np.where(inverse == i)[0] for i in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([buckets[i] for i in chosen]))
    return out


def bootstrap_ci(
    y_true,
    y_pred,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    patients=None,
) -> dict:
    """Bootstrap CIs. If ``patients`` is given, resample patient clusters (not images)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if patients is None:
        draws = [rng.integers(0, n, n) for _ in range(n_boot)]
    else:
        patients = np.asarray(patients)
        if len(patients) != n:
            raise ValueError(f"patients length {len(patients)} != n={n}")
        draws = _patient_bootstrap_indices(rng, patients, n_boot)
    samples = {k: [] for k in _CI_KEYS}
    for idx in draws:
        m = _fast_metrics(y_true[idx], y_pred[idx])
        for k in _CI_KEYS:
            samples[k].append(m[k])
    point = _fast_metrics(y_true, y_pred)
    return {
        k: {
            "point": float(point[k]),
            "lo": float(np.quantile(samples[k], alpha / 2)),
            "hi": float(np.quantile(samples[k], 1 - alpha / 2)),
        }
        for k in _CI_KEYS
    }


def paired_bootstrap_diff(
    y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 0, patients=None
) -> dict:
    """CI of metric(b) - metric(a) on the same images. 'significant' if CI excludes 0.

    If ``patients`` is given, resample patient clusters (paired within each draw).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    pa = np.asarray(pred_a, dtype=np.int64)
    pb = np.asarray(pred_b, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if patients is None:
        draws = [rng.integers(0, n, n) for _ in range(n_boot)]
    else:
        patients = np.asarray(patients)
        if len(patients) != n:
            raise ValueError(f"patients length {len(patients)} != n={n}")
        draws = _patient_bootstrap_indices(rng, patients, n_boot)
    diffs = {k: [] for k in _CI_KEYS}
    for idx in draws:
        ma = _fast_metrics(y_true[idx], pa[idx])
        mb = _fast_metrics(y_true[idx], pb[idx])
        for k in _CI_KEYS:
            diffs[k].append(mb[k] - ma[k])
    pa_m = _fast_metrics(y_true, pa)
    pb_m = _fast_metrics(y_true, pb)
    out = {}
    for k in _CI_KEYS:
        lo, hi = np.quantile(diffs[k], [0.025, 0.975])
        out[k] = {
            "delta": float(pb_m[k] - pa_m[k]),
            "lo": float(lo),
            "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0),
        }
    return out
