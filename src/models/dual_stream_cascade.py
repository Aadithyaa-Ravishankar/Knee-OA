"""
dual_stream_cascade.py
======================
Stage 6 — Dual-stream ResNet-50 with cascade identification.

Technique
---------
1) Binary head: KL-0 (normal) vs OA (KL-1..4)
2) OA CORN head: ordinal grade among {1,2,3,4}  → 3 threshold logits

Shared global+local dual-stream trunk (same as Stage 5).
Inference: if binary says normal → 0; else corn_predict(OA) + 1.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .dual_stream import center_zoom


class DualStreamCascade(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.5,
        local_frac: float = 0.72,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.local_frac = local_frac

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features  # 2048
        fused = feat_dim * 2

        self.dropout = nn.Dropout(p=dropout)
        self.binary_head = nn.Linear(fused, 1)       # logit: P(OA)
        self.oa_corn_head = nn.Linear(fused, 3)      # CORN for grades 1..4

        if freeze_backbone:
            self.freeze_backbone()

    def encode_fused(self, x: torch.Tensor) -> torch.Tensor:
        feat_g = self.features(x).flatten(1)
        feat_l = self.features(center_zoom(x, self.local_frac)).flatten(1)
        return torch.cat([feat_g, feat_l], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        binary_logit : (B, 1)
        oa_logits    : (B, 3)  CORN thresholds for remapped grades 0..3 (= KL 1..4)
        """
        fused = self.dropout(self.encode_fused(x))
        return self.binary_head(fused), self.oa_corn_head(fused)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, oa_threshold: float = 0.5) -> torch.Tensor:
        from src.losses.ordinal_loss import corn_predict

        bin_logit, oa_logits = self.forward(x)
        is_oa = torch.sigmoid(bin_logit.squeeze(1)) >= oa_threshold
        grades = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if is_oa.any():
            grades[is_oa] = corn_predict(oa_logits[is_oa]) + 1
        return grades

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def load_dual_stream_backbone(self, ckpt_path: str | Path, map_location="cpu") -> int:
        """Warm-start trunk from Stage 5 DualStreamResNet50 checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        feat_state = {
            k.replace("features.", ""): v
            for k, v in state.items()
            if k.startswith("features.")
        }
        missing, unexpected = self.features.load_state_dict(feat_state, strict=False)
        n = len(feat_state)
        print(f"  Warm-started backbone from {ckpt_path} ({n} tensors)")
        if missing:
            print(f"  [warn] missing keys: {len(missing)}")
        if unexpected:
            print(f"  [warn] unexpected keys: {len(unexpected)}")
        return n

    @property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"DualStreamCascade(trainable_params={self.n_trainable_params:,}, "
            f"local_frac={self.local_frac})"
        )
