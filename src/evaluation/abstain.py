"""
abstain.py
==========
Ambiguity-aware decisions for KL grading (esp. KL-0 ↔ KL-1).

Uses CORN threshold probabilities:
  - grade = number of thresholds with sigmoid > 0.5
  - abstain when near a boundary with low confidence
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.losses.ordinal_loss import corn_predict, corn_probs


def corn_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Confidence proxy in [0, 1]: how far the decisive threshold is from 0.5,
    combined with peak class probability from corn_probs.
    """
    probs = corn_probs(logits)  # (B, 5)
    peak, _ = probs.max(dim=1)
    # margin of nearest cumulative gate to 0.5
    p_gt = torch.sigmoid(logits)
    gate_margin = (p_gt - 0.5).abs().min(dim=1).values  # in [0, 0.5]
    conf = 0.5 * peak + 0.5 * (gate_margin / 0.5).clamp(0, 1)
    return conf


def predict_with_abstain(
    logits: torch.Tensor,
    conf_threshold: float = 0.55,
    abstain_labels: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    preds     : int grades (abstained samples keep the raw pred for analysis)
    abstain   : bool mask
    conf      : confidence scores
    """
    if abstain_labels is None:
        abstain_labels = {0, 1}  # only abstain on early-grade ambiguity by default

    preds = corn_predict(logits)
    conf = corn_confidence(logits)
    abstain = (conf < conf_threshold) & torch.tensor(
        [int(p) in abstain_labels for p in preds.tolist()],
        device=logits.device,
    )
    # Also abstain if top-2 mass on {0,1} is high and nearly tied
    probs = corn_probs(logits)
    p01 = probs[:, 0] + probs[:, 1]
    tie01 = (probs[:, 0] - probs[:, 1]).abs() < 0.12
    abstain = abstain | ((p01 > 0.7) & tie01)

    return (
        preds.detach().cpu().numpy(),
        abstain.detach().cpu().numpy().astype(bool),
        conf.detach().cpu().numpy(),
    )


def metrics_with_abstain(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abstain: np.ndarray,
) -> dict[str, float]:
    """Coverage + QWK/acc on non-abstained subset."""
    from src.evaluation.metrics import quadratic_weighted_kappa, overall_accuracy

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    keep = ~np.asarray(abstain).astype(bool)
    coverage = float(keep.mean()) if len(keep) else 0.0
    if keep.sum() == 0:
        return {"coverage": 0.0, "qwk_covered": 0.0, "acc_covered": 0.0}
    return {
        "coverage": coverage,
        "qwk_covered": quadratic_weighted_kappa(y_true[keep], y_pred[keep]),
        "acc_covered": overall_accuracy(y_true[keep], y_pred[keep]),
        "n_abstain": float((~keep).sum()),
        "n_covered": float(keep.sum()),
    }
