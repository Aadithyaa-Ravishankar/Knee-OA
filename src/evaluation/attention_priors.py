"""
Soft spatial priors for Stage-19 joint-focus training.

Encourages feature-map energy in the central joint band and discourages energy
on the image frame border — without hard-masking pixels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_energy(act: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) → (B,1,H,W) non-negative energy map."""
    return act.float().pow(2).mean(dim=1, keepdim=True)


def _box_mass(energy: torch.Tensor, top: float, bot: float, left: float, right: float) -> torch.Tensor:
    """Mean over batch of (energy in box / total energy). energy: (B,1,H,W)."""
    _b, _c, h, w = energy.shape
    t, b = int(h * top), int(h * bot)
    l, r = int(w * left), int(w * right)
    t, b = max(0, min(t, h - 1)), max(t + 1, min(b, h))
    l, r = max(0, min(l, w - 1)), max(l + 1, min(r, w))
    total = energy.sum(dim=(2, 3)) + 1e-8
    inside = energy[:, :, t:b, l:r].sum(dim=(2, 3))
    return (inside / total).mean()


def joint_band_mass(
    energy: torch.Tensor,
    top_frac: float = 0.30,
    bottom_frac: float = 0.70,
    left_frac: float = 0.15,
    right_frac: float = 0.85,
) -> torch.Tensor:
    return _box_mass(energy, top_frac, bottom_frac, left_frac, right_frac)


def frame_border_mass(energy: torch.Tensor, border_frac: float = 0.08) -> torch.Tensor:
    """Fraction of energy in the outer frame rim (1 - inner box mass)."""
    border_frac = float(min(max(border_frac, 0.02), 0.25))
    inner = _box_mass(energy, border_frac, 1.0 - border_frac, border_frac, 1.0 - border_frac)
    return 1.0 - inner


def attention_prior_loss(
    acts: list[torch.Tensor],
    *,
    border_frac: float = 0.08,
    band_weight: float = 1.0,
    border_weight: float = 1.0,
    include_local: bool = True,
) -> torch.Tensor:
    """Soft loss on global (and optional local) stream feature energy.

    acts[0] = global map; acts[1] = local (center dual) when present.
    L = band_weight * (1 - joint_band) + border_weight * frame_border
    Set include_local=False to steer only the rim/global stream.
    """
    if not acts:
        raise ValueError("acts must be non-empty")
    terms = []
    # Global: fight frame-edge shortcuts
    eg = feature_energy(acts[0])
    terms.append(band_weight * (1.0 - joint_band_mass(eg)))
    terms.append(border_weight * frame_border_mass(eg, border_frac=border_frac))
    # Local crop is already joint-centered; optional extra band reward inside the crop
    if include_local and len(acts) > 1:
        el = feature_energy(acts[1])
        terms.append(0.5 * band_weight * (1.0 - joint_band_mass(el)))
    return torch.stack(terms).sum()


@torch.no_grad()
def border_mass_numpy(cam, border_frac: float = 0.08) -> float:
    """Eval helper for CAM arrays (H,W) in any scale."""
    import numpy as np

    cam = np.asarray(cam, dtype=np.float64)
    h, w = cam.shape
    bf = float(min(max(border_frac, 0.02), 0.25))
    t, b = int(h * bf), int(h * (1.0 - bf))
    l, r = int(w * bf), int(w * (1.0 - bf))
    total = float(cam.sum()) + 1e-8
    inner = float(cam[t:b, l:r].sum())
    return 1.0 - inner / total
