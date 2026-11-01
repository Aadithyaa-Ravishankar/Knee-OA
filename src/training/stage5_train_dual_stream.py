"""
stage5_train_dual_stream.py
===========================
Stage 5 — Dual-stream ResNet-50 + CORN, aligned to Stage 4a hparams.

P0  : Stage 4a champion frozen (reports/champion_model.json)
P1  : Dual-stream (global + joint center-zoom)  ← only architecture change
P4  : G&P sampler down-weight (gp_sample_scale=0.5)

Training knobs match Stage 4a exactly:
  LR=5e-5 (head & backbone), WD=5e-4, dropout=0.5, warmup=3, no Mixup
  patience=7 (shorter wall time)

Usage:
    python src/training/stage5_train_dual_stream.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataloader import build_dataloaders, summarize_loaders
from src.models.dual_stream import DualStreamResNet50
from src.losses.ordinal_loss import CORNLoss, corn_predict
from src.evaluation.metrics import compute_all_metrics, print_metrics, print_confusion_matrix
from src.evaluation.abstain import predict_with_abstain, metrics_with_abstain

DATASET_DIR = PROJECT_ROOT / "Dataset_combined"
MODELS_DIR  = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "stage5"
FIGURES_DIR = REPORTS_DIR / "figures"
STATS_JSON  = PROJECT_ROOT / "reports" / "stage2b" / "normalization_stats_combined.json"
CHAMPION_JSON = PROJECT_ROOT / "reports" / "champion_model.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16  # dual-stream ≈2× VRAM; 32 OOM on 4GB GPUs
NUM_WORKERS = 0
EPOCHS = 50
# Exact Stage 4a optimizer / regularizer (single LR after unfreeze)
LR = 5e-5
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5
LOCAL_FRAC = 0.72
PATIENCE = 7
WARMUP_FREEZE_EPOCHS = 3
GP_SAMPLE_SCALE = 0.5       # P4 only extra data knob
NUM_CLASSES = 5
CORN_OUT = 4

MEAN = [0.578707, 0.578707, 0.578707]
STD = [0.145419, 0.145419, 0.145419]

CHECKPOINT = MODELS_DIR / "resnet50_dual_stream_corn_best.pth"
HISTORY_JSON = REPORTS_DIR / "training_history_dual_stream.json"


def load_norm_stats() -> None:
    global MEAN, STD
    if STATS_JSON.exists():
        stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
        MEAN, STD = stats["mean_3ch"], stats["std_3ch"]
        print(f"  Loaded norm stats  mean={MEAN}  std={STD}")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, n_seen = 0.0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  train", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)  # hard CORN targets — no Mixup (4a-aligned)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n_seen += bs
        all_preds.extend(corn_predict(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_all_metrics(all_labels, all_preds, prefix="train_")
    metrics["train_loss"] = running_loss / max(n_seen, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val_", with_abstain=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_abstain = []

    for imgs, labels in tqdm(loader, desc=f"  {prefix.strip('_')}", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        if with_abstain:
            preds, abstain, _ = predict_with_abstain(logits)
            all_abstain.extend(abstain.tolist())
        else:
            preds = corn_predict(logits).cpu().numpy()
        all_preds.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_all_metrics(all_labels, all_preds, prefix=prefix)
    metrics[f"{prefix}loss"] = running_loss / len(loader.dataset)
    if with_abstain:
        ab = metrics_with_abstain(
            np.array(all_labels), np.array(all_preds), np.array(all_abstain)
        )
        metrics.update({f"{prefix}{k}": v for k, v in ab.items()})
    return metrics, all_labels, all_preds


def plot_curves(history, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Stage 5 — Dual-stream CORN", fontsize=13, fontweight="bold")
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_qwk"], label="Train QWK")
    axes[1].plot(epochs, history["val_qwk"], label="Val QWK")
    axes[1].axhline(max(history["val_qwk"]), ls="--", color="grey", alpha=0.5)
    axes[1].set_title("QWK"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "training_curves_dual_stream.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_cm(y_true, y_pred, out_dir: Path, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    labels = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix (Stage5 dual) — {split}", fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=1)
    axes[0].set_title("Raw"); axes[1].set_title("Row-normalized")
    plt.tight_layout()
    out = out_dir / f"confusion_matrix_dual_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def preflight() -> bool:
    print("\n[0/7] Preflight...")
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    check("champion_pinned", CHAMPION_JSON.exists(), str(CHAMPION_JSON))
    champ_ckpt = MODELS_DIR / "CHAMPION_stage4a_corn.pth"
    check("champion_ckpt", champ_ckpt.exists(), str(champ_ckpt))
    check("dataset", DATASET_DIR.exists(), str(DATASET_DIR))
    train = DATASET_DIR / "train"
    total = sum(len(list((train / str(g)).glob("*.png"))) for g in range(5))
    check("train_6437", total == 6437, f"total={total}")
    return ok


def main() -> None:
    set_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )

    print("\n" + "=" * 60)
    print("  STAGE 5 — DUAL-STREAM CORN (4a hparams + dual + G&P×0.5)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(
        f"  local_frac={LOCAL_FRAC}  LR={LR}  WD={WEIGHT_DECAY}  "
        f"dropout={DROPOUT}  mixup=OFF  gp_scale={GP_SAMPLE_SCALE}  patience={PATIENCE}"
    )
    if CHAMPION_JSON.exists():
        champ = json.loads(CHAMPION_JSON.read_text(encoding="utf-8"))
        print(f"  P0 champion test QWK = {champ.get('test_qwk')} (frozen, not overwritten)")

    if not preflight():
        sys.exit(1)
    load_norm_stats()

    print("\n[1/7] DataLoaders (G&P reweight P4)...")
    loaders = build_dataloaders(
        root_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        channels=3,
        mean=MEAN,
        std=STD,
        use_weighted_sampler=True,
        gp_sample_scale=GP_SAMPLE_SCALE,
        pin_memory=device.type == "cuda",
    )
    summarize_loaders(loaders)

    print("\n[2/7] Dual-stream ResNet-50...")
    model = DualStreamResNet50(
        num_classes=CORN_OUT,
        pretrained=True,
        dropout=DROPOUT,
        local_frac=LOCAL_FRAC,
        freeze_backbone=True,
    ).to(device)
    print(f"  {model}")

    criterion = CORNLoss(num_classes=NUM_CLASSES)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    history = {
        "train_loss": [], "val_loss": [],
        "train_qwk": [], "val_qwk": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "lr": [],
    }
    best_val_qwk, best_epoch, patience_counter = -1.0, 0, 0
    t0 = time.time()

    print("\n[3/7] Training...\n")
    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_FREEZE_EPOCHS + 1:
            print("  → Unfreeze full backbone (4a-style single LR)")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
            print(f"  {model}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:>3}/{EPOCHS}  (lr={current_lr:.2e})")

        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_m, _, _ = evaluate(model, loaders["val"], criterion, device, prefix="val_")

        for k in ("loss", "qwk", "macro_f1"):
            history[f"train_{k}"].append(train_m[f"train_{k}"])
            history[f"val_{k}"].append(val_m[f"val_{k}"])
        history["lr"].append(current_lr)

        print(
            f"  train | loss={train_m['train_loss']:.4f} qwk={train_m['train_qwk']:.4f} "
            f"f1={train_m['train_macro_f1']:.4f}"
        )
        print(
            f"  val   | loss={val_m['val_loss']:.4f} qwk={val_m['val_qwk']:.4f} "
            f"f1={val_m['val_macro_f1']:.4f}"
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
                    "stage": "5",
                    "ordinal": "CORN",
                    "hparams": {
                        "local_frac": LOCAL_FRAC,
                        "mixup": False,
                        "gp_sample_scale": GP_SAMPLE_SCALE,
                        "lr": LR,
                        "dropout": DROPOUT,
                        "weight_decay": WEIGHT_DECAY,
                        "warmup_freeze_epochs": WARMUP_FREEZE_EPOCHS,
                        "mean": MEAN,
                        "std": STD,
                        "aligned_to": "stage4a",
                    },
                },
                CHECKPOINT,
            )
            print(f"  ✅ New best val QWK={best_val_qwk:.4f}")
        else:
            patience_counter += 1
            print(f"  patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. Best={best_epoch}")
                break

    elapsed = time.time() - t0
    HISTORY_JSON.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n  Training time: {elapsed/60:.1f} min")

    print("\n[4/7] Curves...")
    plot_curves(history, FIGURES_DIR)

    print("\n[5/7] Test + abstain (ambiguity gate)...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = DualStreamResNet50(
        num_classes=CORN_OUT, pretrained=False, dropout=DROPOUT, local_frac=LOCAL_FRAC
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch {ckpt['epoch']}")

    test_m, y_true, y_pred = evaluate(
        model, loaders["test"], criterion, device, prefix="test_", with_abstain=True
    )
    print_metrics(test_m, title="Test (dual-stream)")
    print_confusion_matrix(y_true, y_pred)
    plot_cm(y_true, y_pred, FIGURES_DIR, "test")

    val_m, yv, pv = evaluate(model, loaders["val"], criterion, device, prefix="val_", with_abstain=True)
    plot_cm(yv, pv, FIGURES_DIR, "val")

    print("\n[6/7] Compare to champion (P0)...")
    champ_qwk = 0.8321
    if CHAMPION_JSON.exists():
        champ_qwk = float(json.loads(CHAMPION_JSON.read_text(encoding="utf-8"))["test_qwk"])
    delta = test_m["test_qwk"] - champ_qwk
    print(f"  Champion 4a test QWK: {champ_qwk:.4f}")
    print(f"  Stage5    test QWK: {test_m['test_qwk']:.4f}  (Δ={delta:+.4f})")
    if "test_coverage" in test_m:
        print(
            f"  Abstain coverage={test_m['test_coverage']:.3f}  "
            f"QWK_covered={test_m.get('test_qwk_covered', float('nan')):.4f}"
        )

    print("\n[7/7] Summary...")
    summary = {
        "stage": "5",
        "model": "DualStreamResNet50 + CORN",
        "implements": [
            "P0-champion-ref",
            "P1-dual-stream",
            "P4-gp-reweight",
            "abstain",
            "4a-exact-hparams-no-mixup",
        ],
        "best_epoch": best_epoch,
        "training_time_minutes": round(elapsed / 60, 2),
        "best_val_qwk": round(best_val_qwk, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_m.items()},
        "champion_test_qwk": champ_qwk,
        "delta_vs_champion": round(delta, 4),
        "checkpoint": str(CHECKPOINT),
        "hparams": {
            "local_frac": LOCAL_FRAC,
            "mixup": False,
            "gp_sample_scale": GP_SAMPLE_SCALE,
            "lr": LR,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "warmup_freeze_epochs": WARMUP_FREEZE_EPOCHS,
            "patience": PATIENCE,
            "mean": MEAN,
            "std": STD,
            "aligned_to": "stage4a",
        },
    }
    out = REPORTS_DIR / "stage5_dual_stream_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n" + "=" * 60)
    print("  STAGE 5 COMPLETE")
    print(f"  Best val QWK : {best_val_qwk:.4f} (epoch {best_epoch})")
    print(f"  Test QWK     : {test_m['test_qwk']:.4f}  vs champion {champ_qwk:.4f}")
    print("  Next: python src/training/stage5_gradcam_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
