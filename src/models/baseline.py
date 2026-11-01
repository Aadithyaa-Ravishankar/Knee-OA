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
    Single-stream ResNet-50 with a custom classification head.

    Parameters
    ----------
    num_classes : 5 for standard CE (KL-0..4); use 4 for CORN ordinal head
    pretrained  : load ImageNet weights (strongly recommended)
    dropout     : dropout probability before the final linear layer
    freeze_backbone : if True, freeze all layers except the final FC head
    """

    def __init__(
        self,
        num_classes:      int  = 5,
        pretrained:       bool = True,
        dropout:          float = 0.3,
        freeze_backbone:  bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original FC layer; keep everything up to the global pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features    # 2048 for ResNet-50

        # ── Classification / ordinal head ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

        # ResNet children inside Sequential:
        # 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4 layer1, 5 layer2, 6 layer3, 7 layer4, 8 avgpool
        self._layer3_idx = 6

        # ── Optional backbone freeze ──────────────────────────────────────────
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224) — 3-channel X-ray tensor
        Returns logits of shape (B, num_classes)
          - CE mode: num_classes=5
          - CORN mode: num_classes=4 (thresholds)
        """
        feat = self.features(x)        # (B, 2048, 1, 1)
        return self.classifier(feat)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (head still trains)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (call after warm-up phase)."""
        for param in self.features.parameters():
            param.requires_grad = True

    def unfreeze_from_layer3(self) -> None:
        """
        Anti-overfit partial unfreeze: keep stem/layer1/layer2 frozen;
        train layer3 + layer4 (+ avgpool) and the classifier.
        """
        for idx, child in enumerate(self.features.children()):
            requires = idx >= self._layer3_idx
            for param in child.parameters():
                param.requires_grad = requires

    def trainable_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> list[dict]:
        """AdamW param groups with differential LR (backbone vs head)."""
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]
        groups = []
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay}
            )
        if head_params:
            groups.append(
                {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
            )
        return groups

    @property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"BaselineResNet50("
            f"trainable_params={self.n_trainable_params:,})"
        )
