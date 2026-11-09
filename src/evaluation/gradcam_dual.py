"""
gradcam_dual.py
===============
Correct Grad-CAM for dual-stream models that run one shared trunk twice
(full image + center zoom).

Hooking the shared trunk module (the old approach) overwrites activations and
gradients on the second call, so it mixes the zoomed pass's activations with
the other pass's gradients and paints the zoomed map over the full frame.

Here each stream's layer4 map is taken as an explicit tensor, gradients are
taken with autograd.grad per stream, and the zoomed CAM is pasted back into
the center box it actually covers.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.ensemble import s6_class_probs, s6_predict_with_threshold
from src.models.dual_stream import center_zoom


class LegacyCascadeAdapter:
    """Adapter for src.models.dual_stream_cascade.DualStreamCascade."""

    def __init__(self, model, oa_threshold: float = 0.5):
        self.model = model
        self.oa_threshold = oa_threshold
        self.trunk = model.features[:8]
        self.pool = model.features[8]

    def maps(self, x: torch.Tensor) -> list[tuple[torch.Tensor, float | None]]:
        a_g = self.trunk(x)
        a_l = self.trunk(center_zoom(x, self.model.local_frac))
        return [(a_g, None), (a_l, self.model.local_frac)]

    def outputs_from_maps(self, acts: list[torch.Tensor]):
        fused = torch.cat([self.pool(a).flatten(1) for a in acts], dim=1)
        fused = self.model.dropout(fused)
        bin_l = self.model.binary_head(fused)
        oa_l = self.model.oa_corn_head(fused)
        probs = s6_class_probs(bin_l, oa_l)
        pred = s6_predict_with_threshold(bin_l, oa_l, self.oa_threshold)
        return probs, pred


class OrdinalNetAdapter:
    """Adapter for src.models.ordinal_net.OrdinalNet (Pipeline).

    Center dual (champion) needs no side. Medlat can pass side=0/1 later.
    Grad-CAM uses the KL head only (aux heads ignored).
    """

    def __init__(self, model, side: torch.Tensor | None = None):
        self.model = model
        self.side = side

    def maps(self, x):
        return self.model.feature_maps(x, self.side)

    def outputs_from_maps(self, acts):
        raw = self.model.forward_from_maps(acts)
        return self.model.to_probs(raw), self.model.decode(raw)


def _paste_center(cam: torch.Tensor, frac: float, out_hw: tuple[int, int]) -> torch.Tensor:
    """cam (1,1,h,w) in zoomed coords → (1,1,H,W) in full-image coords (zeros outside the box)."""
    h, w = out_hw
    frac = float(min(max(frac, 0.4), 0.95))
    ch, cw = int(round(h * frac)), int(round(w * frac))
    top, left = (h - ch) // 2, (w - cw) // 2
    small = F.interpolate(cam, size=(ch, cw), mode="bilinear", align_corners=False)
    canvas = cam.new_zeros(1, 1, h, w)
    canvas[:, :, top : top + ch, left : left + cw] = small
    return canvas


def _norm01(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def dual_gradcam(adapter, x: torch.Tensor, target_class: int | None = None) -> dict:
    """
    x : (1, 3, H, W). Model must be in eval mode.
    Returns per-stream and combined CAMs in full-image coordinates, all in [0, 1],
    plus the raw (un-normalized) combined map for energy comparisons.
    """
    h, w = x.shape[-2:]
    with torch.enable_grad():
        stream = adapter.maps(x)
        acts = [a for a, _ in stream]
        probs, pred = adapter.outputs_from_maps(acts)
        tgt = int(pred[0].item()) if target_class is None else int(target_class)
        score = probs[0, tgt]
        grads = torch.autograd.grad(score, acts)

    cams_full = []
    for (a, frac), g in zip(stream, grads):
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True)).detach()
        if frac is None:
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        else:
            cam = _paste_center(cam, frac, (h, w))
        cams_full.append(cam[0, 0].cpu().numpy())

    combined_raw = np.sum(cams_full, axis=0)
    out = {
        "target_class": tgt,
        "pred_class": int(pred[0].item()),
        "probs": probs[0].detach().cpu().numpy(),
        "combined": _norm01(combined_raw.copy()),
        "combined_raw": combined_raw,
        "streams": [_norm01(c.copy()) for c in cams_full],
    }
    return out


def band_mass(cam: np.ndarray, top_frac: float = 0.30, bottom_frac: float = 0.70,
              left_frac: float = 0.15, right_frac: float = 0.85) -> float:
    """Share of CAM energy in the central joint band of a pre-cropped knee image."""
    h, w = cam.shape
    t, b = int(h * top_frac), int(h * bottom_frac)
    l, r = int(w * left_frac), int(w * right_frac)
    total = float(cam.sum()) + 1e-8
    return float(cam[t:b, l:r].sum()) / total


def tensor_to_gray_u8(
    x: torch.Tensor,
    mean: list[float] | tuple[float, ...] = (0.578707, 0.578707, 0.578707),
    std: list[float] | tuple[float, ...] = (0.145419, 0.145419, 0.145419),
) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) normalized tensor -> HxW uint8 grayscale."""
    if x.dim() == 4:
        x = x[0]
    g = x[0].detach().float().cpu().numpy() * float(std[0]) + float(mean[0])
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


def overlay_cam(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a [0,1] CAM onto a grayscale or BGR image.

    Returns BGR uint8 suitable for cv2.imwrite.
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.max() > 1.0 or cam.min() < 0.0:
        cam = _norm01(cam)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        base = image.copy()
    else:
        raise ValueError(f"expected HxW or HxWx3 image, got shape {image.shape}")
    if heat.shape[:2] != base.shape[:2]:
        heat = cv2.resize(heat, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(base, 1.0 - a, heat, a, 0)
