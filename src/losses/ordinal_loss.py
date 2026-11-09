"""
ordinal_loss.py
===============
CORN-style ordinal loss for KL grades 0..4.

Uses K-1 = 4 cumulative binary tasks:
  task k predicts P(grade > k), k = 0..3

Far mistakes (e.g. 0 vs 4) activate more failed tasks → higher loss.
Keeps the ResNet-50 backbone; only the final Linear has 4 outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORNLoss(nn.Module):
    """
    Cumulative ordinal binary cross-entropy (CORN / ordinal CE family).

    logits : (B, num_classes - 1)
    labels : (B,) integer grades in [0, num_classes - 1]
    """

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = num_classes
        self.n_thresholds = num_classes - 1

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        labels       : (B,) hard integer grades (ignored if soft_targets given)
        soft_targets : optional (B, K-1) soft P(Y > k) for Mixup / label smoothing
        """
        if logits.shape[1] != self.n_thresholds:
            raise ValueError(
                f"Expected logits with {self.n_thresholds} cols, got {logits.shape[1]}"
            )
        if soft_targets is not None:
            if soft_targets.shape != logits.shape:
                raise ValueError(
                    f"soft_targets shape {tuple(soft_targets.shape)} != "
                    f"logits shape {tuple(logits.shape)}"
                )
            targets = soft_targets.clamp(0.0, 1.0)
        else:
            labels = labels.long()
            targets = torch.stack(
                [(labels > k).float() for k in range(self.n_thresholds)],
                dim=1,
            )
        losses = [
            F.binary_cross_entropy_with_logits(logits[:, k], targets[:, k], reduction="mean")
            for k in range(self.n_thresholds)
        ]
        return torch.stack(losses).mean()


class WeightedCORNLoss(nn.Module):
    """
    CORN with per-threshold weights.

    For OA remapped grades 0..3 (= KL 1..4), thresholds are:
      t0: >0  (KL1 | KL2+)   — often down-weighted (ambiguous 1↔2)
      t1: >1  (KL2 | KL3+)   — protect severe entry
      t2: >2  (KL3 | KL4)    — protect KL-3/4 split
    """

    def __init__(
        self,
        num_classes: int = 4,
        threshold_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = num_classes
        self.n_thresholds = num_classes - 1
        if threshold_weights is None:
            threshold_weights = tuple(1.0 for _ in range(self.n_thresholds))
        if len(threshold_weights) != self.n_thresholds:
            raise ValueError(
                f"threshold_weights length {len(threshold_weights)} != {self.n_thresholds}"
            )
        self.register_buffer(
            "thr_w",
            torch.tensor(threshold_weights, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape[1] != self.n_thresholds:
            raise ValueError(
                f"Expected logits with {self.n_thresholds} cols, got {logits.shape[1]}"
            )
        if soft_targets is not None:
            targets = soft_targets.clamp(0.0, 1.0)
        else:
            labels = labels.long()
            targets = torch.stack(
                [(labels > k).float() for k in range(self.n_thresholds)],
                dim=1,
            )
        losses = torch.stack(
            [
                F.binary_cross_entropy_with_logits(
                    logits[:, k], targets[:, k], reduction="mean"
                )
                for k in range(self.n_thresholds)
            ]
        )
        w = self.thr_w.to(device=losses.device, dtype=losses.dtype)
        return (losses * w).sum() / w.sum().clamp(min=1e-8)


def corn_threshold_targets(labels: torch.Tensor, n_thresholds: int = 4) -> torch.Tensor:
    """Hard CORN targets (B, K-1): 1 if label > k else 0."""
    labels = labels.long()
    return torch.stack([(labels > k).float() for k in range(n_thresholds)], dim=1)


def mixup_corn_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    prob: float = 0.5,
    n_thresholds: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mixup for CORN: mix images and soft threshold targets.

    Returns
    -------
    mixed_images, labels_a (for metric logging), soft_targets (B, K-1)
    """
    targets_a = corn_threshold_targets(labels, n_thresholds)
    if alpha <= 0.0 or prob <= 0.0 or images.size(0) < 2:
        return images, labels, targets_a

    if torch.rand(1).item() > prob:
        return images, labels, targets_a

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    # Keep mix informative (avoid near-0/1 which collapses to identity)
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    soft = lam * targets_a + (1.0 - lam) * targets_a[index]
    return mixed, labels, soft


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN logits (B, K-1) → predicted grade (B,).
    Count how many thresholds fire (sigmoid > 0.5).
    """
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def corn_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Approximate class probabilities (B, K) from cumulative threshold probs.
    Useful for logging; argmax may differ slightly from corn_predict.
    """
    # P(Y > k)
    p_gt = torch.sigmoid(logits)  # (B, K-1)
    # P(Y >= 0) = 1
    # P(Y = 0) = 1 - P(Y > 0)
    # P(Y = k) = P(Y > k-1) - P(Y > k)
    # P(Y = K-1) = P(Y > K-2)
    parts = [1.0 - p_gt[:, 0]]
    for k in range(1, logits.shape[1]):
        parts.append(p_gt[:, k - 1] - p_gt[:, k])
    parts.append(p_gt[:, -1])
    probs = torch.stack(parts, dim=1).clamp(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs
