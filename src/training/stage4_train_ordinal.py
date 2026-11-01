"""
stage4_train_ordinal.py
=======================
Stage 4a — Same ResNet-50 backbone + CORN ordinal loss + anti-overfit.

Improvements vs Stage 3 combined+aug:
  - CORN ordinal loss (4 threshold logits; far mistakes cost more)
  - Dataset_combined includes G&P KL-2/3/4
  - Stronger augmentation
  - Higher dropout / weight decay, lower LR
  - Freeze backbone for a short warm-up, then unfreeze

Usage:
    python src/training/stage4_train_ordinal.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataloader import build_dataloaders, summarize_loaders
from src.models.baseline import BaselineResNet50
from src.losses.ordinal_loss import CORNLoss, corn_predict
from src.evaluation.metrics import (
    compute_all_metrics, print_metrics, print_confusion_matrix
)

DATASET_DIR = PROJECT_ROOT / "Dataset_combined"
MODELS_DIR  = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "stage4"
FIGURES_DIR = REPORTS_DIR / "figures"
STATS_JSON  = PROJECT_ROOT / "reports" / "stage2b" / "normalization_stats_combined.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
BATCH_SIZE   = 32
NUM_WORKERS  = 0
EPOCHS       = 50
LR           = 5e-5
WEIGHT_DECAY = 5e-4
DROPOUT      = 0.5
PATIENCE     = 10
WARMUP_FREEZE_EPOCHS = 3
NUM_CLASSES  = 5
CORN_OUT     = NUM_CLASSES - 1  # 4 thresholds

# Fallback = previous Chen stats; overwritten if STATS_JSON exists
MEAN = [0.578707, 0.578707, 0.578707]
STD  = [0.127051, 0.127051, 0.127051]

CHECKPOINT   = MODELS_DIR / "resnet50_ordinal_corn_best.pth"
HISTORY_JSON = REPORTS_DIR / "training_history_ordinal.json"


def load_norm_stats() -> None:
    global MEAN, STD
    if STATS_JSON.exists():
        stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
        MEAN = stats["mean_3ch"]
        STD = stats["std_3ch"]
        print(f"  Loaded norm stats from {STATS_JSON}")
        print(f"  mean={MEAN}  std={STD}")
    else:
        print("  [WARN] Combined stats JSON missing — using Chen fallback mean/std")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  train", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = corn_predict(logits).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_all_metrics(all_labels, all_preds, prefix="train_")
    metrics["train_loss"] = running_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val_"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=f"  {prefix.strip('_')}", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = corn_predict(logits).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_all_metrics(all_labels, all_preds, prefix=prefix)
    metrics[f"{prefix}loss"] = running_loss / len(loader.dataset)
    return metrics, all_labels, all_preds


def plot_curves(history: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Stage 4 — ResNet-50 + CORN Ordinal", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#4C72B0")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#DD8452")
    axes[0].set_title("CORN Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_qwk"], label="Train QWK", color="#4C72B0")
    axes[1].plot(epochs, history["val_qwk"], label="Val QWK", color="#DD8452")
    axes[1].axhline(y=max(history["val_qwk"]), linestyle="--", color="grey", alpha=0.5,
                     label=f"Best={max(history['val_qwk']):.4f}")
    axes[1].set_title("QWK"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["train_macro_f1"], label="Train F1", color="#4C72B0")
    axes[2].plot(epochs, history["val_macro_f1"], label="Val F1", color="#DD8452")
    axes[2].set_title("Macro-F1"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "training_curves_ordinal.png"
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
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix (Ordinal) — {split}", fontsize=13, fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0], linewidths=0.5)
    axes[0].set_title("Raw"); axes[0].set_xlabel("Pred"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Row-normalized"); axes[1].set_xlabel("Pred"); axes[1].set_ylabel("True")
    plt.tight_layout()
    out = out_dir / f"confusion_matrix_ordinal_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def preflight_gates() -> bool:
    """Fail fast if combined dataset / expected counts look wrong."""
    print("\n[0/6] Preflight gates...")
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        ok = ok and cond

    check("dataset_exists", DATASET_DIR.exists(), str(DATASET_DIR))
    train = DATASET_DIR / "train"
    counts = {g: len(list((train / str(g)).glob("*.png"))) for g in range(5)}
    check("train_total_6437", sum(counts.values()) == 6437, f"counts={counts}")
    check("has_gp_kl2", len(list((train / "2").glob("gp_*.png"))) == 232, "KL-2 G&P")
    check("val_intact", len(list((DATASET_DIR / "val").rglob("gp_*.png"))) == 0, "no gp in val")
    check("test_intact", len(list((DATASET_DIR / "test").rglob("gp_*.png"))) == 0, "no gp in test")
    check("ordinal_loss_import", True, "CORNLoss ready")
    return ok


def main() -> None:
    set_seed(SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("  STAGE 4a — ResNet-50 + CORN ORDINAL")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  LR={LR}  WD={WEIGHT_DECAY}  dropout={DROPOUT}  freeze_warmup={WARMUP_FREEZE_EPOCHS}")

    if not preflight_gates():
        print("\n[FAIL] Preflight gates failed — rebuild Dataset_combined first.")
        sys.exit(1)

    load_norm_stats()

    print("\n[1/6] Building DataLoaders...")
    loaders = build_dataloaders(
        root_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        channels=3,
        mean=MEAN,
        std=STD,
        use_weighted_sampler=True,
        splits=("train", "val", "test"),
        pin_memory=device.type == "cuda",
    )
    summarize_loaders(loaders)
    print(f"  Class weights: {loaders['train'].dataset.class_weights().numpy().round(3)}")

    print("\n[2/6] Initializing ResNet-50 (CORN head = 4 logits)...")
    model = BaselineResNet50(
        num_classes=CORN_OUT,
        pretrained=True,
        dropout=DROPOUT,
        freeze_backbone=True,
    ).to(device)
    print(f"  {model}")

    criterion = CORNLoss(num_classes=NUM_CLASSES)
    # Only head trains during freeze warm-up
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    print("\n[3/6] Training...\n")
    history = {
        "train_loss": [], "val_loss": [],
        "train_qwk": [], "val_qwk": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "train_accuracy": [], "val_accuracy": [],
        "lr": [],
    }
    best_val_qwk = -1.0
    patience_counter = 0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_FREEZE_EPOCHS + 1:
            print("  → Unfreezing backbone for full fine-tune")
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

        current_lr = optimizer.param_groups[0]["lr"]
        frozen = not next(model.features.parameters()).requires_grad
        print(f"\nEpoch {epoch:>3}/{EPOCHS}  (lr={current_lr:.2e}, backbone_frozen={frozen})")

        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_m, _, _ = evaluate(model, loaders["val"], criterion, device, prefix="val_")

        for key in ("loss", "qwk", "macro_f1", "accuracy"):
            history[f"train_{key}"].append(train_m[f"train_{key}"])
            history[f"val_{key}"].append(val_m[f"val_{key}"])
        history["lr"].append(current_lr)

        print(
            f"  train | loss={train_m['train_loss']:.4f}  "
            f"qwk={train_m['train_qwk']:.4f}  f1={train_m['train_macro_f1']:.4f}"
        )
        print(
            f"  val   | loss={val_m['val_loss']:.4f}  "
            f"qwk={val_m['val_qwk']:.4f}  f1={val_m['val_macro_f1']:.4f}"
        )

        scheduler.step(val_m["val_qwk"])

        if val_m["val_qwk"] > best_val_qwk:
            best_val_qwk = val_m["val_qwk"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_qwk": best_val_qwk,
                    "ordinal": "CORN",
                    "hparams": {
                        "lr": LR, "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
                        "mean": MEAN, "std": STD,
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
    HISTORY_JSON.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n[4/6] Curves...")
    plot_curves(history, FIGURES_DIR)

    print("\n[5/6] Test evaluation (best checkpoint)...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    # Ensure backbone unfrozen structure matches saved weights
    model = BaselineResNet50(
        num_classes=CORN_OUT, pretrained=False, dropout=DROPOUT, freeze_backbone=False
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch {ckpt['epoch']}")

    test_m, y_true, y_pred = evaluate(
        model, loaders["test"], criterion, device, prefix="test_"
    )
    print_metrics(test_m, title="Test Set Evaluation (Ordinal)")
    print_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, FIGURES_DIR, split="test")

    val_m_final, y_true_v, y_pred_v = evaluate(
        model, loaders["val"], criterion, device, prefix="val_"
    )
    plot_confusion_matrix(y_true_v, y_pred_v, FIGURES_DIR, split="val")

    print("\n[6/6] Summary...")
    summary = {
        "stage": "4a",
        "model": "BaselineResNet50 + CORN",
        "backbone": "ImageNet ResNet-50 (unchanged family)",
        "best_epoch": best_epoch,
        "training_time_minutes": round(elapsed / 60, 2),
        "best_val_qwk": round(best_val_qwk, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_m.items()},
        "checkpoint": str(CHECKPOINT),
        "compare_to_stage3_combined_aug_test_qwk": 0.8155,
        "hparams": {
            "lr": LR, "batch_size": BATCH_SIZE, "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY, "warmup_freeze_epochs": WARMUP_FREEZE_EPOCHS,
            "mean": MEAN, "std": STD,
        },
    }
    summary_path = REPORTS_DIR / "stage4_ordinal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved → {summary_path}")

    print("\n" + "=" * 60)
    print("  STAGE 4a COMPLETE")
    print(f"  Best val QWK : {best_val_qwk:.4f}  (epoch {best_epoch})")
    print(f"  Test QWK     : {test_m['test_qwk']:.4f}")
    print(f"  Test Macro-F1: {test_m['test_macro_f1']:.4f}")
    print(f"  Test Accuracy: {test_m['test_accuracy']:.4f}")
    print(f"  (Stage3 combined+aug test QWK was 0.8155)")
    print("=" * 60)


if __name__ == "__main__":
    main()
