"""
transforms.py
=============
Image transforms for the Knee OA X-ray pipeline.

Key design decisions:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to
  improve bone/cartilage contrast in X-ray images before any normalization.
- Images remain grayscale (1 channel) throughout.
- Mean/std normalization uses values computed from the training split
  (see compute_stats.py). Fallback defaults are provided.
- A 3-channel "repeat" variant is included for ImageNet-pretrained backbones.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────────────────────────────────────
# CLAHE helper (PIL → PIL, grayscale)
# ──────────────────────────────────────────────────────────────────────────────

class ApplyCLAHE:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a
    grayscale PIL image.

    Parameters
    ----------
    clip_limit : CLAHE clip limit (default 2.0 is conservative for X-rays)
    tile_size  : tile grid size for CLAHE (default (8, 8))

    Note: cv2.CLAHE is NOT picklable, so we store the parameters and
    re-create the object lazily on first use (safe for DataLoader workers).
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_size  = tile_size
        self._clahe     = None   # lazy-initialized per worker process

    def _get_clahe(self):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_size,
            )
        return self._clahe

    def __call__(self, img: Image.Image) -> Image.Image:
        # PIL → numpy uint8
        arr = np.array(img.convert("L"), dtype=np.uint8)
        enhanced = self._get_clahe().apply(arr)
        return Image.fromarray(enhanced, mode="L")

    def __repr__(self) -> str:
        return f"ApplyCLAHE(clip_limit={self.clip_limit}, tile_size={self.tile_size})"


# ──────────────────────────────────────────────────────────────────────────────
# Repeat-channels helper (1-ch → 3-ch for ImageNet backbones)
# ──────────────────────────────────────────────────────────────────────────────

class GrayscaleToRGB:
    """Repeat a single-channel tensor to 3 channels without color conversion."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)
        return tensor

    def __repr__(self) -> str:
        return "GrayscaleToRGB()"


# ──────────────────────────────────────────────────────────────────────────────
# Pre-computed normalization stats (from compute_stats.py)
# Update these values after running compute_stats.py on the training split.
# ──────────────────────────────────────────────────────────────────────────────

# Grayscale (1-channel) — from Dataset_combined train (compute_stats.py)
TRAIN_MEAN_1CH = [0.578707]
TRAIN_STD_1CH  = [0.145419]

# 3-channel repeat (same value on all channels)
TRAIN_MEAN_3CH = [0.578707, 0.578707, 0.578707]
TRAIN_STD_3CH  = [0.145419, 0.145419, 0.145419]


# ──────────────────────────────────────────────────────────────────────────────
# Transform factories
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms_1ch(
    mean: list[float] = TRAIN_MEAN_1CH,
    std:  list[float] = TRAIN_STD_1CH,
) -> T.Compose:
    """
    Training transforms — grayscale, 1-channel output.

    Pipeline:
        CLAHE → light spatial/intensity aug → ToTensor → Normalize
    """
    return T.Compose([
        ApplyCLAHE(clip_limit=2.0, tile_size=(8, 8)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms_1ch(
    mean: list[float] = TRAIN_MEAN_1CH,
    std:  list[float] = TRAIN_STD_1CH,
) -> T.Compose:
    """
    Evaluation transforms (val / test) — identical pipeline, no augmentation.
    """
    return T.Compose([
        ApplyCLAHE(clip_limit=2.0, tile_size=(8, 8)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_train_transforms_3ch(
    mean: list[float] = TRAIN_MEAN_3CH,
    std:  list[float] = TRAIN_STD_3CH,
) -> T.Compose:
    """
    Training transforms — 3-channel output (for ImageNet-pretrained ResNet-50).

    On-the-fly aug (train only): flip, small rotation, brightness/contrast.
    Eval transforms stay unaugmented.
    """
    return T.Compose([
        ApplyCLAHE(clip_limit=2.0, tile_size=(8, 8)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        GrayscaleToRGB(),
        T.Normalize(mean=mean, std=std),
        # Mild occlusion regularizer (after normalize; fill with 0 == mean)
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
    ])


def get_eval_transforms_3ch(
    mean: list[float] = TRAIN_MEAN_3CH,
    std:  list[float] = TRAIN_STD_3CH,
) -> T.Compose:
    """
    Evaluation transforms — 3-channel output.
    """
    return T.Compose([
        ApplyCLAHE(clip_limit=2.0, tile_size=(8, 8)),
        T.ToTensor(),
        GrayscaleToRGB(),
        T.Normalize(mean=mean, std=std),
    ])
