"""
dual_stream.py
==============
Dual-stream ResNet-50 for KL grading (Stage 5 / P1).

Streams
-------
  global : full 224×224 X-ray
  local  : center-crop zoom on the joint (default 72% FOV) → resized to 224

Shared ImageNet ResNet-50 backbone → concat 4096-d → Dropout → CORN head (4 logits).
Same backbone family as Stage 3/4; adds joint focus to cut mid-grade ambiguity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def center_zoom(x: torch.Tensor, frac: float = 0.72) -> torch.Tensor:
    """
    Center-crop a fraction of the spatial FOV, then bilinear-resize to original HxW.
    x : (B, C, H, W) or (C, H, W)
    """
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    b, c, h, w = x.shape
    frac = float(min(max(frac, 0.4), 0.95))
    ch, cw = int(round(h * frac)), int(round(w * frac))
    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = x[:, :, top : top + ch, left : left + cw]
    out = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
    return out.squeeze(0) if squeeze else out


class DualStreamResNet50(nn.Module):
    """
    Shared ResNet-50 trunk with global + local (joint zoom) streams.

    Parameters
    ----------
    num_classes : 4 for CORN, 5 for CE
    local_frac  : center-crop fraction for the local stream
    dropout     : head dropout
    freeze_backbone : freeze trunk at init
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.45,
        local_frac: float = 0.72,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.local_frac = local_frac

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        feat_dim = backbone.fc.in_features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim * 2, num_classes),
        )

        self._layer3_idx = 6
        if freeze_backbone:
            self.freeze_backbone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)  # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224)
        returns logits (B, num_classes)
        """
        feat_g = self.encode(x)
        local = center_zoom(x, self.local_frac)
        feat_l = self.encode(local)
        return self.classifier(torch.cat([feat_g, feat_l], dim=1))

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def unfreeze_from_layer3(self) -> None:
        for idx, child in enumerate(self.features.children()):
            req = idx >= self._layer3_idx
            for p in child.parameters():
                p.requires_grad = req

    def trainable_param_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> list[dict]:
        backbone = [p for p in self.features.parameters() if p.requires_grad]
        head = [p for p in self.classifier.parameters() if p.requires_grad]
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay})
        if head:
            groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
        return groups

    @property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"DualStreamResNet50(trainable_params={self.n_trainable_params:,}, local_frac={self.local_frac})"
