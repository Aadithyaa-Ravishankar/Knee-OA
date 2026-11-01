"""
dataset.py
==========
PyTorch Dataset for the Knee OA KL-grade dataset.

Directory layout expected:
    <root>/
        train/  0/  1/  2/  3/  4/
        val/    0/  1/  2/  3/  4/
        test/   0/  1/  2/  3/  4/
        auto_test/ 0/ 1/ 2/ 3/ 4/

Returns:
    image  : torch.Tensor (C, H, W)  — after applying `transform`
    label  : int                     — KL grade (0-4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


VALID_SPLITS = ("train", "val", "test", "auto_test")
KL_GRADES    = (0, 1, 2, 3, 4)
IMG_EXTS     = {".png", ".jpg", ".jpeg"}


class KneeOADataset(Dataset):
    """
    Reads pre-split Knee OA X-ray images and their KL grade labels.

    Parameters
    ----------
    root_dir  : path to the dataset root containing train/val/test sub-dirs
    split     : one of 'train', 'val', 'test', 'auto_test'
    transform : torchvision transform or callable applied to the PIL image
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        assert split in VALID_SPLITS, f"split must be one of {VALID_SPLITS}"

        self.root_dir  = Path(root_dir)
        self.split     = split
        self.transform = transform

        self.samples: list[Tuple[Path, int]] = []
        self._load_samples()

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _load_samples(self) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for grade in KL_GRADES:
            grade_dir = split_dir / str(grade)
            if not grade_dir.exists():
                continue
            for fpath in sorted(grade_dir.iterdir()):
                if fpath.suffix.lower() in IMG_EXTS:
                    self.samples.append((fpath, grade))

        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir}")

    # ─────────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("L")   # grayscale

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    # ─────────────────────────────────────────────
    # Utility: class weights for imbalanced training
    # ─────────────────────────────────────────────

    def class_counts(self) -> dict[int, int]:
        """Return {grade: count} for this split."""
        counts: dict[int, int] = {g: 0 for g in KL_GRADES}
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for each class.
        Shape: (5,) — one weight per KL grade.
        Use with nn.CrossEntropyLoss(weight=...) or WeightedRandomSampler.
        """
        counts = self.class_counts()
        total  = sum(counts.values())
        n_cls  = len(KL_GRADES)
        weights = torch.tensor(
            [total / (n_cls * max(counts[g], 1)) for g in KL_GRADES],
            dtype=torch.float32,
        )
        return weights

    def sample_weights(
        self,
        gp_scale: float = 1.0,
        class_scale: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """
        Per-sample weights for WeightedRandomSampler.
        Shape: (len(dataset),)

        gp_scale    : down-weight G&P files (name starts with 'gp_'), e.g. 0.5
        class_scale : extra multipliers per KL grade after inverse-frequency.
                      Example cascade balance (undersample KL-0, protect KL-1):
                        {0: 0.45, 1: 1.35, 2: 0.75, 3: 1.0, 4: 1.15}
        """
        cls_w = self.class_weights()
        scales = class_scale or {}
        weights = []
        for fpath, label in self.samples:
            w = cls_w[label].item() * float(scales.get(label, 1.0))
            if gp_scale != 1.0 and fpath.name.lower().startswith("gp_"):
                w *= gp_scale
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        counts = self.class_counts()
        lines  = [f"KneeOADataset(split='{self.split}', total={len(self)})"]
        for g in KL_GRADES:
            lines.append(f"  KL-{g}: {counts[g]:>5}")
        return "\n".join(lines)
