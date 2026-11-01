"""
dataloader.py
=============
DataLoader factory for the Knee OA KL-grade dataset.

Usage
-----
from src.preprocessing.dataloader import build_dataloaders

loaders = build_dataloaders(
    root_dir   = "/path/to/dataset",
    batch_size = 32,
    num_workers= 4,
    channels   = 3,   # 1 or 3
    mean       = [0.485, 0.456, 0.406],  # from compute_stats.py
    std        = [0.229, 0.224, 0.225],
)
train_loader = loaders["train"]
val_loader   = loaders["val"]
test_loader  = loaders["test"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset    import KneeOADataset
from .transforms import (
    get_train_transforms_1ch, get_eval_transforms_1ch,
    get_train_transforms_3ch, get_eval_transforms_3ch,
)


def build_dataloaders(
    root_dir:    str | Path,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    channels:    int  = 3,
    mean:        Optional[List[float]] = None,
    std:         Optional[List[float]] = None,
    use_weighted_sampler: bool = True,
    splits: tuple = ("train", "val", "test"),
    pin_memory: bool = True,
    gp_sample_scale: float = 1.0,
    class_sample_scale: Optional[dict] = None,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    root_dir              : dataset root (contains train/, val/, test/ subdirs)
    batch_size            : images per batch
    num_workers           : parallel data-loading workers
    channels              : 1 (grayscale) or 3 (RGB repeat for ResNet)
    mean / std            : normalization stats — use values from compute_stats.py
    use_weighted_sampler  : if True, apply WeightedRandomSampler on the train split
    splits                : which splits to build loaders for
    pin_memory            : pin CPU memory for faster GPU transfer
    gp_sample_scale       : down-weight G&P train samples in the sampler (P4), e.g. 0.5
    class_sample_scale    : optional per-KL multipliers (e.g. undersample KL-0)

    Returns
    -------
    dict mapping split name → DataLoader
    """
    # ── Select transforms ────────────────────────────────────────────────────
    if channels == 1:
        kw = dict(mean=mean or [0.5], std=std or [0.5])
        train_tf = get_train_transforms_1ch(**kw)
        eval_tf  = get_eval_transforms_1ch(**kw)
    else:
        kw = dict(
            mean=mean or [0.5, 0.5, 0.5],
            std =std  or [0.5, 0.5, 0.5],
        )
        train_tf = get_train_transforms_3ch(**kw)
        eval_tf  = get_eval_transforms_3ch(**kw)

    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        is_train = (split == "train")
        tf       = train_tf if is_train else eval_tf

        dataset = KneeOADataset(root_dir=root_dir, split=split, transform=tf)

        if is_train and use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights     = dataset.sample_weights(
                    gp_scale=gp_sample_scale,
                    class_scale=class_sample_scale,
                ),
                num_samples = len(dataset),
                replacement = True,
            )
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                sampler     = sampler,        # mutually exclusive with shuffle
                num_workers = num_workers,
                pin_memory  = pin_memory,
                drop_last   = True,           # avoid single-sample batches
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = pin_memory,
                drop_last   = False,
            )

        loaders[split] = loader

    return loaders


def summarize_loaders(loaders: Dict[str, DataLoader]) -> None:
    """Print a summary table of all DataLoaders."""
    print(f"{'Split':<12} {'Batches':>8} {'Images':>8} {'Batch Size':>11}")
    print("-" * 45)
    for split, loader in loaders.items():
        n_imgs    = len(loader.dataset)   # type: ignore[arg-type]
        n_batches = len(loader)
        bs        = loader.batch_size or "?"
        print(f"{split:<12} {n_batches:>8} {n_imgs:>8} {bs:>11}")
