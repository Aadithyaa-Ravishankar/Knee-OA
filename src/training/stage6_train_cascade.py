"""
stage6_train_cascade.py
=======================
Stage 6 — Dual-stream + cascade identification.

Solves (targeted):
  - Misclassification of early grades via 0-vs-OA then OA ordinal grade
  - Class imbalance via sampler: undersample KL-0, protect/boost KL-1, mild KL-2 cut
  - Overfitting via Stage-4a-style knobs (dropout 0.5, WD 5e-4, warmup, patience 7)
  - Keeps dual-stream joint focus from Stage 5
  - Warm-starts backbone from Stage 5 champion when available

Usage:
    python src/training/stage6_train_cascade.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataloader import build_dataloaders, summarize_loaders
from src.models.dual_stream_cascade import DualStreamCascade
from src.losses.ordinal_loss import CORNLoss, corn_predict
from src.evaluation.metrics import compute_all_metrics, print_metrics, print_confusion_matrix
from src.evaluation.abstain import metrics_with_abstain

DATASET_DIR = PROJECT_ROOT / "Dataset_combined"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "stage6"
FIGURES_DIR = REPORTS_DIR / "figures"
STATS_JSON = PROJECT_ROOT / "reports" / "stage2b" / "normalization_stats_combined.json"
CHAMPION_JSON = PROJECT_ROOT / "reports" / "champion_model.json"
STAGE5_CKPT = MODELS_DIR / "resnet50_dual_stream_corn_best.pth"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16          # dual-stream on 4GB GPU
NUM_WORKERS = 0
EPOCHS = 50
LR = 5e-5
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5
LOCAL_FRAC = 0.72
PATIENCE = 7
WARMUP_FREEZE_EPOCHS = 3
GP_SAMPLE_SCALE = 0.5
# Undersample KL-0, protect KL-1, mild cut KL-2 (no file deletion)
CLASS_SAMPLE_SCALE = {0: 0.45, 1: 1.35, 2: 0.75, 3: 1.0, 4: 1.15}
BIN_LOSS_WEIGHT = 1.0
OA_LOSS_WEIGHT = 1.0
OA_THRESH = 0.5

MEAN = [0.578707, 0.578707, 0.578707]
STD = [0.145419, 0.145419, 0.145419]

CHECKPOINT = MODELS_DIR / "resnet50_dual_cascade_best.pth"
HISTORY_JSON = REPORTS_DIR / "training_history_cascade.json"


def load_norm_stats() -> None:
    global MEAN, STD
    if STATS_JSON.exists():
        stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
        MEAN, STD = stats["mean_3ch"], stats["std_3ch"]
        print(f"  Loaded norm stats mean={MEAN} std={STD}")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cascade_predict(bin_logit: torch.Tensor, oa_logits: torch.Tensor, thresh: float = 0.5):
    """Return grades (B,), OA probs (B,), binary confidence margin."""
    p_oa = torch.sigmoid(bin_logit.squeeze(1))
    is_oa = p_oa >= thresh
    grades = torch.zeros(bin_logit.size(0), dtype=torch.long, device=bin_logit.device)
    if is_oa.any():
        grades[is_oa] = corn_predict(oa_logits[is_oa]) + 1
    # confidence: distance of p_oa from 0.5
    conf = (p_oa - 0.5).abs() * 2.0
    return grades, p_oa, conf


def cascade_loss(bin_logit, oa_logits, labels, corn_crit):
    """Binary BCE on all + CORN on OA samples only."""
    y_oa = (labels > 0).float()
    loss_bin = F.binary_cross_entropy_with_logits(bin_logit.squeeze(1), y_oa)

    oa_mask = labels > 0
    if oa_mask.any():
        oa_labels = labels[oa_mask] - 1  # map KL1..4 → 0..3
        loss_oa = corn_crit(oa_logits[oa_mask], oa_labels)
    else:
        loss_oa = bin_logit.new_zeros(())

    return BIN_LOSS_WEIGHT * loss_bin + OA_LOSS_WEIGHT * loss_oa, loss_bin.detach(), loss_oa.detach()


def train_one_epoch(model, loader, corn_crit, optimizer, device):
    model.train()
    running, n_seen = 0.0, 0
    preds, labels_all = [], []

    for imgs, labels in tqdm(loader, desc="  train", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        bin_logit, oa_logits = model(imgs)
        loss, _, _ = cascade_loss(bin_logit, oa_logits, labels, corn_crit)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = imgs.size(0)
        running += loss.item() * bs
        n_seen += bs
        g, _, _ = cascade_predict(bin_logit.detach(), oa_logits.detach(), OA_THRESH)
        preds.extend(g.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    metrics = compute_all_metrics(labels_all, preds, prefix="train_")
    metrics["train_loss"] = running / max(n_seen, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, corn_crit, device, prefix="val_", with_abstain=False):
    model.eval()
    running, n_seen = 0.0, 0
    preds, labels_all, abstain_flags = [], [], []

    for imgs, labels in tqdm(loader, desc=f"  {prefix.strip('_')}", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bin_logit, oa_logits = model(imgs)
        loss, _, _ = cascade_loss(bin_logit, oa_logits, labels, corn_crit)
        running += loss.item() * imgs.size(0)
        n_seen += imgs.size(0)

        g, p_oa, conf = cascade_predict(bin_logit, oa_logits, OA_THRESH)
        preds.extend(g.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        if with_abstain:
            # Abstain on fuzzy normal/OA boundary or low-conf early OA grade
            fuzzy_bin = conf < 0.25
            early = (g <= 1)
            abstain_flags.extend((fuzzy_bin & early).cpu().numpy().tolist())

    metrics = compute_all_metrics(labels_all, preds, prefix=prefix)
    metrics[f"{prefix}loss"] = running / max(n_seen, 1)
    if with_abstain:
        ab = metrics_with_abstain(
            np.array(labels_all), np.array(preds), np.array(abstain_flags)
        )
        metrics.update({f"{prefix}{k}": v for k, v in ab.items()})
    return metrics, labels_all, preds


def plot_curves(history, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Stage 6 — Dual-stream Cascade", fontweight="bold")
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_qwk"], label="Train QWK")
    axes[1].plot(epochs, history["val_qwk"], label="Val QWK")
    axes[1].axhline(max(history["val_qwk"]), ls="--", color="grey", alpha=0.5)
    axes[1].set_title("QWK"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "training_curves_cascade.png"
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
    fig.suptitle(f"Confusion Matrix (Stage6 cascade) — {split}", fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=1)
    axes[0].set_title("Raw"); axes[1].set_title("Row-normalized")
    plt.tight_layout()
    out = out_dir / f"confusion_matrix_cascade_{split}.png"
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

    check("dataset", DATASET_DIR.exists(), str(DATASET_DIR))
    total = sum(len(list((DATASET_DIR / "train" / str(g)).glob("*.png"))) for g in range(5))
    check("train_6437", total == 6437, f"total={total}")
    check("stage5_optional", True, f"warmstart={'yes' if STAGE5_CKPT.exists() else 'no (ImageNet only)'}")
    return ok


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  STAGE 6 — DUAL-STREAM CASCADE (0 vs OA → grade)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(
        f"  LR={LR} WD={WEIGHT_DECAY} dropout={DROPOUT} batch={BATCH_SIZE} "
        f"patience={PATIENCE} gp_scale={GP_SAMPLE_SCALE}"
    )
    print(f"  class_scale={CLASS_SAMPLE_SCALE}")

    champ_qwk = 0.8359
    if CHAMPION_JSON.exists():
        champ = json.loads(CHAMPION_JSON.read_text(encoding="utf-8"))
        champ_qwk = float(champ.get("test_qwk", champ_qwk))
        print(f"  Current champion test QWK = {champ_qwk}")

    if not preflight():
        sys.exit(1)
    load_norm_stats()

    print("\n[1/7] DataLoaders (imbalance-aware sampler)...")
    loaders = build_dataloaders(
        root_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        channels=3,
        mean=MEAN,
        std=STD,
        use_weighted_sampler=True,
        gp_sample_scale=GP_SAMPLE_SCALE,
        class_sample_scale=CLASS_SAMPLE_SCALE,
        pin_memory=device.type == "cuda",
    )
    summarize_loaders(loaders)

    print("\n[2/7] DualStreamCascade...")
    model = DualStreamCascade(
        pretrained=True,
        dropout=DROPOUT,
        local_frac=LOCAL_FRAC,
        freeze_backbone=True,
    ).to(device)
    if STAGE5_CKPT.exists():
        model.load_dual_stream_backbone(STAGE5_CKPT, map_location="cpu")
        model.to(device)
    print(f"  {model}")

    corn_crit = CORNLoss(num_classes=4)  # OA grades 1..4 → 4 classes, 3 thresholds
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
            print("  → Unfreeze backbone")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
            print(f"  {model}")

        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:>3}/{EPOCHS}  (lr={lr:.2e})")

        train_m = train_one_epoch(model, loaders["train"], corn_crit, optimizer, device)
        val_m, _, _ = evaluate(model, loaders["val"], corn_crit, device, prefix="val_")

        for k in ("loss", "qwk", "macro_f1"):
            history[f"train_{k}"].append(train_m[f"train_{k}"])
            history[f"val_{k}"].append(val_m[f"val_{k}"])
        history["lr"].append(lr)

        gap = train_m["train_qwk"] - val_m["val_qwk"]
        print(
            f"  train | loss={train_m['train_loss']:.4f} qwk={train_m['train_qwk']:.4f} "
            f"f1={train_m['train_macro_f1']:.4f}"
        )
        print(
            f"  val   | loss={val_m['val_loss']:.4f} qwk={val_m['val_qwk']:.4f} "
            f"f1={val_m['val_macro_f1']:.4f}  gap={gap:.4f}"
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
                    "train_qwk": train_m["train_qwk"],
                    "qwk_gap": gap,
                    "stage": "6",
                    "technique": "dual_stream_cascade",
                    "hparams": {
                        "lr": LR,
                        "dropout": DROPOUT,
                        "weight_decay": WEIGHT_DECAY,
                        "batch_size": BATCH_SIZE,
                        "local_frac": LOCAL_FRAC,
                        "gp_sample_scale": GP_SAMPLE_SCALE,
                        "class_sample_scale": CLASS_SAMPLE_SCALE,
                        "mean": MEAN,
                        "std": STD,
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

    print("\n[5/7] Test + abstain...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = DualStreamCascade(
        pretrained=False, dropout=DROPOUT, local_frac=LOCAL_FRAC
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch {ckpt['epoch']}")

    test_m, y_true, y_pred = evaluate(
        model, loaders["test"], corn_crit, device, prefix="test_", with_abstain=True
    )
    print_metrics(test_m, title="Test (cascade)")
    print_confusion_matrix(y_true, y_pred)
    plot_cm(y_true, y_pred, FIGURES_DIR, "test")
    val_m, yv, pv = evaluate(model, loaders["val"], corn_crit, device, prefix="val_", with_abstain=True)
    plot_cm(yv, pv, FIGURES_DIR, "val")

    delta = test_m["test_qwk"] - champ_qwk
    print("\n[6/7] vs champion...")
    print(f"  Champion QWK: {champ_qwk:.4f}")
    print(f"  Stage6  QWK: {test_m['test_qwk']:.4f}  (Δ={delta:+.4f})")

    print("\n[7/7] Summary...")
    summary = {
        "stage": "6",
        "model": "DualStreamCascade",
        "technique": "0-vs-OA then OA-CORN",
        "dual_stream": True,
        "best_epoch": best_epoch,
        "training_time_minutes": round(elapsed / 60, 2),
        "best_val_qwk": round(best_val_qwk, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_m.items()},
        "champion_test_qwk": champ_qwk,
        "delta_vs_champion": round(delta, 4),
        "checkpoint": str(CHECKPOINT),
        "hparams": {
            "lr": LR,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "warmup_freeze_epochs": WARMUP_FREEZE_EPOCHS,
            "gp_sample_scale": GP_SAMPLE_SCALE,
            "class_sample_scale": CLASS_SAMPLE_SCALE,
            "local_frac": LOCAL_FRAC,
            "mean": MEAN,
            "std": STD,
        },
    }
    out = REPORTS_DIR / "stage6_cascade_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved → {out}")

    # Promote champion if improved
    if delta > 0:
        dest = MODELS_DIR / "CHAMPION_stage6_cascade.pth"
        dest.write_bytes(CHECKPOINT.read_bytes())
        champ_doc = {
            "status": "champion",
            "stage": "6",
            "reason": "Best test QWK (dual-stream cascade)",
            "test_qwk": round(test_m["test_qwk"], 4),
            "best_val_qwk": round(best_val_qwk, 4),
            "checkpoint": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "previous_champion_qwk": champ_qwk,
        }
        CHAMPION_JSON.write_text(json.dumps(champ_doc, indent=2), encoding="utf-8")
        print(f"  ✅ New champion pinned → {dest}")

    print("\n" + "=" * 60)
    print("  STAGE 6 COMPLETE")
    print(f"  Best val QWK : {best_val_qwk:.4f} (epoch {best_epoch})")
    print(f"  Test QWK     : {test_m['test_qwk']:.4f}  vs champ {champ_qwk:.4f}")
    print(f"  Test Acc     : {test_m['test_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
