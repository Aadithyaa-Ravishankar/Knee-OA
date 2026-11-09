"""
gradcam.py
==========
Grad-CAM for ResNet-50 / DualStreamResNet50 (Stage 5 / P3).

Highlights joint-attention failures on misclassified mid-grades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM on a chosen conv module (typically layer4)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        x : (1, 3, H, W)
        returns heatmap (H, W) in [0, 1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            # For CORN: use sum of threshold logits as severity score
            score = logits.sum()
        else:
            score = logits[0, class_idx]
        score.backward()

        assert self.activations is not None and self.gradients is not None
        # Grad-CAM weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_cam_on_gray(
    image_chw: np.ndarray,
    cam: np.ndarray,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """
    image_chw : normalized tensor as numpy (3, H, W)
    returns BGR uint8 overlay for saving with cv2
    """
    # De-normalize first channel (grayscale repeated)
    gray = image_chw[0] * std[0] + mean[0]
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)


def joint_attention_score(cam: np.ndarray, center_frac: float = 0.5) -> float:
    """
    Fraction of CAM energy inside the central box (proxy for 'looking at the joint').
    Low score on errors → ROI/dual-stream needed.
    """
    h, w = cam.shape
    ch, cw = int(h * center_frac), int(w * center_frac)
    top, left = (h - ch) // 2, (w - cw) // 2
    total = float(cam.sum()) + 1e-8
    center = float(cam[top : top + ch, left : left + cw].sum())
    return center / total


def center_joint_mask(
    height: int,
    width: int,
    center_frac: float = 0.55,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Soft binary mask (1, 1, H, W) for central tibiofemoral joint box."""
    ch = max(1, int(round(height * center_frac)))
    cw = max(1, int(round(width * center_frac)))
    top = (height - ch) // 2
    left = (width - cw) // 2
    mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    mask[:, :, top : top + ch, left : left + cw] = 1.0
    return mask


def differentiable_cam_from_acts(
    activations: torch.Tensor,
    score: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Grad-CAM map (B, 1, h, w) from feature activations + scalar/batch score.
    create_graph=True for training priors; False for cheaper eval measurement.
    """
    grads = torch.autograd.grad(
        score,
        activations,
        retain_graph=True,
        create_graph=create_graph,
        only_inputs=True,
    )[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    # Normalize per-sample to a distribution over space
    flat = cam.flatten(2)
    cam = cam / (flat.sum(dim=-1).view(-1, 1, 1, 1) + 1e-8)
    return cam


def joint_cam_mass(
    cam: torch.Tensor,
    center_frac: float = 0.55,
) -> torch.Tensor:
    """
    Per-sample fraction of CAM mass inside the center joint box.
    cam : (B, 1, h, w) non-negative, preferably normalized
    returns (B,)
    """
    _, _, h, w = cam.shape
    mask = center_joint_mask(h, w, center_frac, device=cam.device, dtype=cam.dtype)
    inside = (cam * mask).sum(dim=(1, 2, 3))
    total = cam.sum(dim=(1, 2, 3)) + 1e-8
    return inside / total


def joint_cam_prior_loss(
    cam: torch.Tensor,
    center_frac: float = 0.55,
) -> torch.Tensor:
    """Loss = 1 - mean(joint CAM mass). Lower when attention is on the joint."""
    return 1.0 - joint_cam_mass(cam, center_frac=center_frac).mean()
