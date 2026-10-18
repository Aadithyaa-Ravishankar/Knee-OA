"""
src/evaluation/metrics.py
=========================
Stage 3 — Evaluation metrics for KL-grade classification.

Metrics:
    - Quadratic Weighted Kappa (QWK)  ← primary metric for ordinal grading
    - Macro-averaged F1-score
    - Per-class accuracy
    - Overall accuracy
    - Confusion matrix

All functions accept numpy arrays or plain Python lists.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


KL_LABELS = [0, 1, 2, 3, 4]
KL_NAMES  = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]


def quadratic_weighted_kappa(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> float:
    """
    Quadratic Weighted Kappa — the standard metric for ordinal grading tasks.
    Penalises predictions proportional to the SQUARE of the ordinal distance.
    Range: [-1, 1] where 1 = perfect, 0 = chance, <0 = worse than chance.
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def macro_f1(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> float:
    """Macro-averaged F1-score (unweighted mean across all 5 classes)."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def overall_accuracy(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> float:
    return float(accuracy_score(y_true, y_pred))


def per_class_accuracy(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> dict[int, float]:
    """Per-class recall (= per-class accuracy for each grade)."""
    cm = confusion_matrix(y_true, y_pred, labels=KL_LABELS)
    per_cls: dict[int, float] = {}
    for i, grade in enumerate(KL_LABELS):
        total = cm[i].sum()
        per_cls[grade] = float(cm[i, i] / total) if total > 0 else 0.0
    return per_cls


def compute_all_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute all metrics at once and return as a flat dict.
    The `prefix` argument is used to namespace keys (e.g. 'val_' or 'test_').
    """
    p = prefix
    metrics: dict[str, float] = {
        f"{p}qwk":      quadratic_weighted_kappa(y_true, y_pred),
        f"{p}macro_f1": macro_f1(y_true, y_pred),
        f"{p}accuracy": overall_accuracy(y_true, y_pred),
    }
    for grade, acc in per_class_accuracy(y_true, y_pred).items():
        metrics[f"{p}acc_kl{grade}"] = acc
    return metrics


def print_metrics(metrics: dict[str, float], title: str = "Metrics") -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'─' * 45}")
    print(f"  {title}")
    print(f"{'─' * 45}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"{'─' * 45}")


def print_confusion_matrix(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> None:
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=KL_LABELS)
    print(f"\n  Confusion Matrix (rows=true, cols=pred)")
    print(f"  {'':6}" + "".join(f"{n:>7}" for n in KL_NAMES))
    for i, name in enumerate(KL_NAMES):
        row = "".join(f"{cm[i, j]:>7}" for j in range(len(KL_LABELS)))
        print(f"  {name:<6}{row}")
