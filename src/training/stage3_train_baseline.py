"""
stage3_train_baseline.py
========================
Stage 3 — Train the baseline ResNet-50 KL-grade classifier.

What this does:
  1. Load train / val datasets with CLAHE + Normalize transforms
  2. Build class-weighted DataLoaders with WeightedRandomSampler
  3. Train ResNet-50 with Cross-Entropy loss + class weights
  4. Early stopping on validation QWK (patience = 10)
  5. Save best model checkpoint to outputs/models/
  6. Plot and save training curves to reports/stage3/
  7. Run final evaluation on the test split

What this does NOT do (later stages):
  - No dual-stream, no ordinal loss, no Grad-CAM, no augmentation

Usage:
    cd knee_oa_xai/
    source .venv/bin/activate
    python src/training/stage3_train_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset    import KneeOADataset
from src.preprocessing.transforms import get_train_transforms_3ch, get_eval_transforms_3ch
from src.preprocessing.dataloader import build_dataloaders, summarize_loaders
from src.models.baseline          import BaselineResNet50
from src.evaluation.metrics       import (
    compute_all_metrics, print_metrics, print_confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("/Users/aadithyaaravishankar/Desktop/KNEE OA/dataset")
MODELS_DIR   = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"  / "stage3"
FIGURES_DIR  = REPORTS_DIR / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 32
NUM_WORKERS   = 0           # 0 = main process (avoids pickling issues on Python 3.14)
EPOCHS        = 50
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
PATIENCE      = 10          # early stopping on val QWK
MEAN          = [0.578707, 0.578707, 0.578707]
STD           = [0.127051, 0.127051, 0.127051]

CHECKPOINT    = MODELS_DIR / "baseline_resnet50_best.pth"
HISTORY_JSON  = REPORTS_DIR / "training_history.json"


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  train", leave=False, ncols=80):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_all_metrics(all_labels, all_preds, prefix="train_")
    metrics["train_loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
    prefix:    str = "val_",
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=f"  {prefix.strip('_')}", leave=False, ncols=80):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_all_metrics(all_labels, all_preds, prefix=prefix)
    metrics[f"{prefix}loss"] = epoch_loss
    return metrics, all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Stage 3 — Baseline ResNet-50 Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0")
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#DD8452")
    ax.set_title("Cross-Entropy Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # QWK
    ax = axes[1]
    ax.plot(epochs, history["train_qwk"], label="Train QWK", color="#4C72B0")
    ax.plot(epochs, history["val_qwk"],   label="Val QWK",   color="#DD8452")
    ax.axhline(y=max(history["val_qwk"]), linestyle="--", color="grey", alpha=0.5, label=f"Best={max(history['val_qwk']):.4f}")
    ax.set_title("Quadratic Weighted Kappa")
    ax.set_xlabel("Epoch"); ax.set_ylabel("QWK")
    ax.legend(); ax.grid(alpha=0.3)

    # Macro F1
    ax = axes[2]
    ax.plot(epochs, history["train_macro_f1"], label="Train Macro-F1", color="#4C72B0")
    ax.plot(epochs, history["val_macro_f1"],   label="Val Macro-F1",   color="#DD8452")
    ax.set_title("Macro F1-Score")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_confusion_matrix(y_true, y_pred, out_dir: Path, split: str = "test") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    # Normalize by true class (row-wise)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {split.capitalize()} Split", fontsize=13, fontweight="bold")

    labels = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                linewidths=0.5)
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Row-Normalized (Recall per class)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    out = out_dir / f"confusion_matrix_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  STAGE 3 — BASELINE ResNet-50 TRAINING")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    print(f"  Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    print(f"  Early stop patience : {PATIENCE} epochs (val QWK)")

    # ── Datasets & Loaders ───────────────────────────────────────────────────
    print("\n[1/6] Building DataLoaders...")
    loaders = build_dataloaders(
        root_dir             = DATASET_DIR,
        batch_size           = BATCH_SIZE,
        num_workers          = NUM_WORKERS,
        channels             = 3,
        mean                 = MEAN,
        std                  = STD,
        use_weighted_sampler = True,
        splits               = ("train", "val", "test"),
        pin_memory           = device.type == "cuda",
    )
    summarize_loaders(loaders)

    # Class weights for loss function (from train split)
    train_ds = loaders["train"].dataset
    cls_weights = train_ds.class_weights().to(device)
    print(f"\n  Class weights: {cls_weights.cpu().numpy().round(3)}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[2/6] Initializing model...")
    model = BaselineResNet50(num_classes=5, pretrained=True, dropout=DROPOUT)
    model = model.to(device)
    print(f"  {model}")

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Training loop ────────────────────────────────────────────────────────
    print("\n[3/6] Training...\n")
    history = {
        "train_loss": [], "val_loss": [],
        "train_qwk":  [], "val_qwk":  [],
        "train_macro_f1": [], "val_macro_f1": [],
        "train_accuracy": [], "val_accuracy": [],
        "lr": [],
    }

    best_val_qwk    = -1.0
    patience_counter = 0
    best_epoch      = 0
    t_start         = time.time()

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:>3}/{EPOCHS}  (lr={current_lr:.2e})")

        # Train
        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        # Val
        val_m, _, _ = evaluate(model, loaders["val"], criterion, device, prefix="val_")

        # Record history
        history["train_loss"].append(train_m["train_loss"])
        history["val_loss"].append(val_m["val_loss"])
        history["train_qwk"].append(train_m["train_qwk"])
        history["val_qwk"].append(val_m["val_qwk"])
        history["train_macro_f1"].append(train_m["train_macro_f1"])
        history["val_macro_f1"].append(val_m["val_macro_f1"])
        history["train_accuracy"].append(train_m["train_accuracy"])
        history["val_accuracy"].append(val_m["val_accuracy"])
        history["lr"].append(current_lr)

        print(
            f"  train | loss={train_m['train_loss']:.4f}  "
            f"qwk={train_m['train_qwk']:.4f}  "
            f"f1={train_m['train_macro_f1']:.4f}"
        )
        print(
            f"  val   | loss={val_m['val_loss']:.4f}  "
            f"qwk={val_m['val_qwk']:.4f}  "
            f"f1={val_m['val_macro_f1']:.4f}"
        )

        # LR scheduler step
        scheduler.step(val_m["val_qwk"])

        # Early stopping & best model save
        if val_m["val_qwk"] > best_val_qwk:
            best_val_qwk = val_m["val_qwk"]
            best_epoch   = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_qwk":    best_val_qwk,
                    "hparams": {
                        "lr": LR, "batch_size": BATCH_SIZE,
                        "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
                    },
                },
                CHECKPOINT,
            )
            print(f"  ✅ New best val QWK={best_val_qwk:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            print(f"  patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
                break

    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed/60:.1f} min")

    # Save history
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved → {HISTORY_JSON}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n[4/6] Generating training curves...")
    plot_curves(history, FIGURES_DIR)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n[5/6] Loading best checkpoint for test evaluation...")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")

    test_m, y_true, y_pred = evaluate(
        model, loaders["test"], criterion, device, prefix="test_"
    )
    print_metrics(test_m, title="Test Set Evaluation")
    print_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, FIGURES_DIR, split="test")

    # ── Val evaluation ────────────────────────────────────────────────────────
    val_m_final, y_true_v, y_pred_v = evaluate(
        model, loaders["val"], criterion, device, prefix="val_"
    )
    plot_confusion_matrix(y_true_v, y_pred_v, FIGURES_DIR, split="val")

    # ── Stage 3 summary JSON ──────────────────────────────────────────────────
    print("\n[6/6] Writing Stage 3 summary...")
    summary = {
        "stage": 3,
        "model": "BaselineResNet50",
        "best_epoch": best_epoch,
        "training_time_minutes": round(elapsed / 60, 2),
        "best_val_qwk": round(best_val_qwk, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_m.items()},
        "checkpoint": str(CHECKPOINT),
        "hparams": {
            "lr": LR, "batch_size": BATCH_SIZE,
            "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "mean": MEAN, "std": STD,
        },
    }
    summary_path = REPORTS_DIR / "stage3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {summary_path}")

    print(f"\n{'='*60}")
    print(f"  STAGE 3 COMPLETE")
    print(f"  Best val QWK : {best_val_qwk:.4f}  (epoch {best_epoch})")
    print(f"  Test QWK     : {test_m['test_qwk']:.4f}")
    print(f"  Test Macro-F1: {test_m['test_macro_f1']:.4f}")
    print(f"  Test Accuracy: {test_m['test_accuracy']:.4f}")
    print(f"{'='*60}")
    print(f"\n✅ Stage 3 done. Review reports/stage3/ before proceeding to Stage 4.")


if __name__ == "__main__":
    main()
