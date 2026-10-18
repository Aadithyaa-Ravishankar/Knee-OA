"""
src/models/baseline.py
======================
Stage 3 — Baseline single-stream ResNet-50 classifier.

Architecture:
    ImageNet-pretrained ResNet-50
    → Replace final FC layer with a 5-class head
    → Input: 3-channel 224×224 (grayscale repeated × 3)
    → Output: logits for KL-0 ... KL-4

This is intentionally the simplest possible baseline.
No dual-stream, no ordinal head, no attention — those come in later stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class BaselineResNet50(nn.Module):
    """
    Single-stream ResNet-50 with a custom 5-class classification head.

    Parameters
    ----------
    num_classes : number of output classes (5 for KL-0..KL-4)
    pretrained  : load ImageNet weights (strongly recommended)
    dropout     : dropout probability before the final linear layer
    freeze_backbone : if True, freeze all layers except the final FC head
                      (useful for quick sanity-check / feature-extraction mode)
    """

    def __init__(
        self,
        num_classes:      int  = 5,
        pretrained:       bool = True,
        dropout:          float = 0.3,
        freeze_backbone:  bool = False,
    ) -> None:
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original FC layer; keep everything up to the global pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features    # 2048 for ResNet-50

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

        # ── Optional backbone freeze ──────────────────────────────────────────
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224) — 3-channel X-ray tensor
        Returns logits of shape (B, num_classes)
        """
        feat = self.features(x)        # (B, 2048, 1, 1)
        return self.classifier(feat)   # (B, num_classes)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (call after warm-up phase)."""
        for param in self.features.parameters():
            param.requires_grad = True

    @property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"BaselineResNet50("
            f"trainable_params={self.n_trainable_params:,})"
        )
