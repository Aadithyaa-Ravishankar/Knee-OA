"""
uncertainty.py
==============
Uncertainty quantification and probability calibration (Stage 10):

- MC Dropout predictive distribution (N stochastic forwards with dropout active)
- Predictive entropy
- Expected Calibration Error (ECE)
- Reliability diagram helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep BatchNorm in eval; force Dropout layers into train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Entropy of a categorical distribution.
    probs: (N, C) or (C,) — returns (N,) or scalar.
    """
    p = np.asarray(probs, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    h = -(p * np.log(p)).sum(axis=1)
    return float(h[0]) if single else h


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """
    ECE = sum_m (|B_m|/n) |acc(B_m) - conf(B_m)| over equal-width confidence bins.

    confidences: (N,) in [0, 1] — typically max class probability
    correct: (N,) bool / 0-1 — whether hard prediction matched label
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()
    if len(conf) != len(corr):
        raise ValueError("confidences and correct must have same length")
    n = len(conf)
    if n == 0:
        return {
            "ece": float("nan"),
            "n": 0,
            "n_bins": int(n_bins),
            "bins": [],
        }

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bins = []
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == int(n_bins) - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(corr[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "count": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
            }
        )
    return {"ece": float(ece), "n": n, "n_bins": int(n_bins), "bins": bins}


def reliability_diagram_png(
    ece_report: dict,
    out_path: Path,
    title: str = "Reliability diagram",
) -> Path:
    """Save confidence-vs-accuracy bar/line plot for non-empty bins."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [b for b in ece_report.get("bins", []) if b.get("count", 0) > 0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
    if bins:
        centers = [0.5 * (b["lo"] + b["hi"]) for b in bins]
        widths = [b["hi"] - b["lo"] for b in bins]
        accs = [b["accuracy"] for b in bins]
        confs = [b["confidence"] for b in bins]
        ax.bar(
            centers,
            accs,
            width=[w * 0.9 for w in widths],
            align="center",
            alpha=0.55,
            edgecolor="black",
            label="accuracy",
        )
        ax.plot(centers, confs, marker="o", color="C1", label="confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ece = ece_report.get("ece")
    ax.set_title(f"{title}\nECE = {ece:.4f}" if ece is not None else title)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


@torch.no_grad()
def mc_dropout_probs(
    model: nn.Module,
    x: torch.Tensor,
    side: torch.Tensor | None,
    n_passes: int = 30,
    amp: bool = False,
) -> np.ndarray:
    """
    Run N stochastic forwards with dropout active.

    Returns mean class probs shaped (B, 5).
    """
    enable_mc_dropout(model)
    need_side = side is not None
    acc = None
    for _ in range(int(n_passes)):
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
            raw = model(x, side) if need_side else model(x, None)
            probs = model.to_probs(raw).float()
        acc = probs if acc is None else acc + probs
    mean = acc / float(n_passes)
    model.eval()
    return mean.cpu().numpy()


def _finite_mean(vals: np.ndarray) -> float | None:
    vals = np.asarray(vals, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return float(vals.mean())


def sanitize_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Replace non-finite rows with uniform; renormalize."""
    p = np.asarray(probs, dtype=np.float64).copy()
    if p.ndim == 1:
        p = p[None, :]
    bad = ~np.isfinite(p).all(axis=1)
    if bad.any():
        p[bad] = 1.0 / p.shape[1]
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def summarize_uncertainty(
    probs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Entropy + ECE summary from (N,5) probs and hard labels/preds."""
    probs = sanitize_probs(probs)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    conf = probs[np.arange(len(y_pred)), y_pred]
    correct = y_pred == y_true
    ent = predictive_entropy(probs)
    # ECE uses sanitized confidences
    ece = expected_calibration_error(conf, correct.astype(np.float64), n_bins=n_bins)
    return {
        "n": int(len(y_true)),
        "mean_entropy": _finite_mean(ent),
        "mean_confidence": _finite_mean(conf),
        "accuracy": float(np.mean(correct)),
        "ece": ece["ece"],
        "ece_report": ece,
        "entropy_by_true_kl": {
            str(k): _finite_mean(ent[y_true == k]) if np.any(y_true == k) else None
            for k in range(5)
        },
    }
