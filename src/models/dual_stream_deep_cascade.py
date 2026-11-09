"""
dual_stream_deep_cascade.py
===========================
Stage 8 — Sharper-ROI dual-stream + 3-level cascade.

Cascade
-------
1) OA head:      KL-0 vs OA (1..4)
2) Severity:     mild (1–2) vs severe (3–4)   [OA only]
3a) Mild head:   KL-1 vs KL-2
3b) Severe head: KL-3 vs KL-4

ROI: tighter local center-zoom (default local_frac=0.62) for mid-grade cues.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .dual_stream import center_zoom


class DualStreamDeepCascade(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.5,
        local_frac: float = 0.62,
        local_size: int = 224,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.local_frac = local_frac
        self.local_size = local_size

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        fused = feat_dim * 2

        self.dropout = nn.Dropout(p=dropout)
        self.oa_head = nn.Linear(fused, 1)        # P(OA)
        self.sev_head = nn.Linear(fused, 1)       # P(severe | OA)  grades 3–4
        self.mild_head = nn.Linear(fused, 1)      # P(KL-2 | mild)
        self.severe_head = nn.Linear(fused, 1)    # P(KL-4 | severe)

        if freeze_backbone:
            self.freeze_backbone()

    def _local_view(self, x: torch.Tensor) -> torch.Tensor:
        local = center_zoom(x, self.local_frac)
        if self.local_size != x.shape[-1]:
            local = F.interpolate(
                local, size=(self.local_size, self.local_size),
                mode="bilinear", align_corners=False,
            )
        return local

    def encode_fused(self, x: torch.Tensor) -> torch.Tensor:
        feat_g = self.features(x).flatten(1)
        feat_l = self.features(self._local_view(x)).flatten(1)
        return torch.cat([feat_g, feat_l], dim=1)

    def forward(self, x: torch.Tensor):
        """
        Returns oa_logit, sev_logit, mild_logit, severe_logit — each (B, 1)
        """
        fused = self.dropout(self.encode_fused(x))
        return (
            self.oa_head(fused),
            self.sev_head(fused),
            self.mild_head(fused),
            self.severe_head(fused),
        )

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        oa_thresh: float = 0.5,
        sev_thresh: float = 0.5,
        mild_thresh: float = 0.5,
        severe_thresh: float = 0.5,
    ) -> torch.Tensor:
        oa_l, sev_l, mild_l, severe_l = self.forward(x)
        p_oa = torch.sigmoid(oa_l.squeeze(1))
        p_sev = torch.sigmoid(sev_l.squeeze(1))
        p_g2 = torch.sigmoid(mild_l.squeeze(1))
        p_g4 = torch.sigmoid(severe_l.squeeze(1))

        grades = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        is_oa = p_oa >= oa_thresh
        is_severe = is_oa & (p_sev >= sev_thresh)
        is_mild = is_oa & (~is_severe)

        grades[is_mild] = torch.where(
            p_g2[is_mild] >= mild_thresh,
            torch.tensor(2, device=x.device),
            torch.tensor(1, device=x.device),
        )
        grades[is_severe] = torch.where(
            p_g4[is_severe] >= severe_thresh,
            torch.tensor(4, device=x.device),
            torch.tensor(3, device=x.device),
        )
        return grades

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def load_backbone_from_ckpt(self, ckpt_path: str | Path, map_location="cpu") -> int:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        feat_state = {
            k.replace("features.", ""): v
            for k, v in state.items()
            if k.startswith("features.")
        }
        self.features.load_state_dict(feat_state, strict=False)
        print(f"  Warm-started backbone from {ckpt_path} ({len(feat_state)} tensors)")
        return len(feat_state)

    @property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"DualStreamDeepCascade(trainable={self.n_trainable_params:,}, "
            f"local_frac={self.local_frac}, local_size={self.local_size})"
        )
