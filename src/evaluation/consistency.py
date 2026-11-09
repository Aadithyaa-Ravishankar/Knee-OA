"""
consistency.py
==============
Explanation consistency metrics (Stage 9):

- AFS / band_mass — share of Grad-CAM energy in the joint band
- ESS — SSIM between CAM on original vs mildly perturbed image
- Grade flip rate — fraction of cases whose hard KL grade changes under perturbation

Perturbations are clinically irrelevant: brightness scale and small Gaussian noise
applied in [0, 1] pixel space (denormalize → perturb → renormalize).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.evaluation.gradcam_dual import band_mass


MEAN_3CH = (0.578707, 0.578707, 0.578707)
STD_3CH = (0.145419, 0.145419, 0.145419)


def denormalize(
    x: torch.Tensor,
    mean: Sequence[float] = MEAN_3CH,
    std: Sequence[float] = STD_3CH,
) -> torch.Tensor:
    """Normalized (…,3,H,W) → approximate [0, 1] pixel tensor."""
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    if x.dim() == 3:
        m, s = m[0], s[0]
    return x * s + m


def normalize(
    x: torch.Tensor,
    mean: Sequence[float] = MEAN_3CH,
    std: Sequence[float] = STD_3CH,
) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    if x.dim() == 3:
        m, s = m[0], s[0]
    return (x - m) / s


def perturb_brightness(
    x_norm: torch.Tensor,
    factor: float,
    mean: Sequence[float] = MEAN_3CH,
    std: Sequence[float] = STD_3CH,
) -> torch.Tensor:
    """Scale brightness by `factor` in [0,1] space (e.g. 0.95 / 1.05)."""
    x = denormalize(x_norm, mean, std).clamp(0.0, 1.0)
    x = (x * float(factor)).clamp(0.0, 1.0)
    return normalize(x, mean, std)


def perturb_gaussian_noise(
    x_norm: torch.Tensor,
    sigma: float = 0.01,
    generator: torch.Generator | None = None,
    mean: Sequence[float] = MEAN_3CH,
    std: Sequence[float] = STD_3CH,
) -> torch.Tensor:
    """Add N(0, sigma^2) noise in [0,1] space."""
    x = denormalize(x_norm, mean, std).clamp(0.0, 1.0)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * float(sigma)
    x = (x + noise).clamp(0.0, 1.0)
    return normalize(x, mean, std)


def default_perturbations(
    x_norm: torch.Tensor,
    noise_sigma: float = 0.01,
    seed: int = 0,
) -> list[tuple[str, torch.Tensor]]:
    """Brightness ±5% and one Gaussian noise draw (deterministic given seed)."""
    gen = torch.Generator(device=x_norm.device)
    gen.manual_seed(int(seed))
    return [
        ("brightness_0.95", perturb_brightness(x_norm, 0.95)),
        ("brightness_1.05", perturb_brightness(x_norm, 1.05)),
        ("noise_0.01", perturb_gaussian_noise(x_norm, sigma=noise_sigma, generator=gen)),
    ]


def cam_ssim(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
    """SSIM between two [0,1] heatmaps (ESS building block)."""
    a = np.asarray(cam_a, dtype=np.float64)
    b = np.asarray(cam_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"CAM shape mismatch {a.shape} vs {b.shape}")
    return float(structural_similarity(a, b, data_range=1.0))


def explanation_stability(
    cam_original: np.ndarray,
    cams_perturbed: Iterable[np.ndarray],
) -> float:
    """ESS = mean SSIM(H_orig, H_pert) over perturbations."""
    vals = [cam_ssim(cam_original, c) for c in cams_perturbed]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def grade_flipped(pred_original: int, preds_perturbed: Iterable[int]) -> bool:
    """True if any perturbed hard grade differs from the original prediction."""
    p0 = int(pred_original)
    return any(int(p) != p0 for p in preds_perturbed)


def summarize_consistency(rows: list[dict]) -> dict:
    """Aggregate Stage-9 row dicts into population metrics."""
    if not rows:
        return {
            "n": 0,
            "mean_afs_combined": None,
            "mean_afs_global": None,
            "mean_afs_local": None,
            "frac_afs_gt_0_5": None,
            "mean_ess": None,
            "frac_ess_gt_0_85": None,
            "grade_flip_rate": None,
            "band_mass_uniform_baseline": band_mass(np.ones((224, 224), dtype=np.float64)),
        }

    afs_c = np.array([r["afs_combined"] for r in rows], dtype=np.float64)
    afs_g = np.array([r["afs_global"] for r in rows], dtype=np.float64)
    afs_l = np.array([r["afs_local"] for r in rows if r.get("afs_local") is not None], dtype=np.float64)
    ess = np.array([r["ess"] for r in rows], dtype=np.float64)
    flips = np.array([bool(r["grade_flipped"]) for r in rows], dtype=bool)
    return {
        "n": len(rows),
        "mean_afs_combined": float(afs_c.mean()),
        "mean_afs_global": float(afs_g.mean()),
        "mean_afs_local": float(afs_l.mean()) if len(afs_l) else None,
        "frac_afs_gt_0_5": float((afs_c > 0.5).mean()),
        "mean_ess": float(ess.mean()),
        "frac_ess_gt_0_85": float((ess > 0.85).mean()),
        "grade_flip_rate": float(flips.mean()),
        "band_mass_uniform_baseline": band_mass(np.ones((224, 224), dtype=np.float64)),
        "targets": {
            "afs_gt": 0.5,
            "ess_gt": 0.85,
            "grade_flip_lt": 0.05,
        },
        "pass_afs": bool(float(afs_c.mean()) > 0.5),
        "pass_ess": bool(float(ess.mean()) > 0.85),
        "pass_grade_flip": bool(float(flips.mean()) < 0.05),
    }
