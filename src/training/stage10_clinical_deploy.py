"""
stage10_clinical_deploy.py
==========================
Deployment layer on frozen Stage 6 champion (no weight updates).

1) Load S6 cascade checkpoint
2) Fit temperature T on VAL (NLL of 5-class cascade probs)
3) Sweep abstain + severe-protect on VAL
4) Apply best rules on TEST
5) Write reports/stage10_clinical_deploy.json with two cards:
     - champion: raw S6 full-test metrics (frozen)
     - clinical: calibrate + severe-protect + abstain

Champion JSON is updated ONLY if full-test predictions (severe-protect,
no abstain) clear the promote gate — otherwise left untouched.

Usage:
    python src/training/stage10_clinical_deploy.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataloader import build_dataloaders
from src.models.dual_stream_cascade import DualStreamCascade
from src.evaluation.ensemble import s6_class_probs, s6_predict_with_threshold
from src.evaluation.metrics import compute_all_metrics, print_metrics, per_class_accuracy
from src.evaluation.abstain import metrics_with_abstain

DATASET_DIR = PROJECT_ROOT / "Dataset_combined"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "stage10"
STATS_JSON = PROJECT_ROOT / "reports" / "stage2b" / "normalization_stats_combined.json"
CHAMPION_JSON = PROJECT_ROOT / "reports" / "champion_model.json"
S6_CKPT = MODELS_DIR / "resnet50_dual_cascade_best.pth"
OUT_JSON = REPORTS_DIR / "stage10_clinical_deploy.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FLOOR_QWK = 0.8417
FLOOR_KL1 = 0.55
FLOOR_KL2 = 0.465
FLOOR_KL3 = 0.84
FLOOR_KL4 = 0.84
BATCH_SIZE = 16
MIN_COVERAGE = 0.70          # clinical mode must keep ≥70% answers
OA_THRESH = 0.5


def load_stats():
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    return stats["mean_3ch"], stats["std_3ch"]


@torch.no_grad()
def collect_logits(model, loader, device):
    ys, bins, oas = [], [], []
    model.eval()
    split = getattr(loader.dataset, "split", "?")
    for imgs, labels in tqdm(loader, desc=f"  infer-{split}", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        b, o = model(imgs)
        ys.append(labels.numpy())
        bins.append(b.cpu())
        oas.append(o.cpu())
    return np.concatenate(ys), torch.cat(bins, 0), torch.cat(oas, 0)


def calibrated_probs(bin_logit: torch.Tensor, oa_logits: torch.Tensor, T: float) -> torch.Tensor:
    t = max(float(T), 1e-3)
    return s6_class_probs(bin_logit / t, oa_logits / t)


def fit_temperature(bin_va: torch.Tensor, oa_va: torch.Tensor, y_va: np.ndarray) -> float:
    """LBFGS temperature on VAL NLL (cascade 5-class probs)."""
    y = torch.tensor(y_va, dtype=torch.long)
    log_T = torch.nn.Parameter(torch.zeros(1))  # T = exp(log_T) > 0

    opt = torch.optim.LBFGS([log_T], lr=0.25, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = log_T.exp().clamp(min=0.05, max=10.0)
        probs = s6_class_probs(bin_va / t, oa_va / t).clamp(min=1e-8)
        loss = F.nll_loss(probs.log(), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_T.exp().clamp(min=0.05, max=10.0).item())
    return T


def apply_severe_protect(probs: np.ndarray, preds: np.ndarray, mass_thr: float) -> np.ndarray:
    """
    If model predicts ≤2 but P(KL≥3) is high, upgrade to argmax{3,4}.
    Protects severe pathway without retraining.
    """
    out = preds.copy()
    if mass_thr >= 1.0:
        return out
    p_sev = probs[:, 3] + probs[:, 4]
    mask = (out <= 2) & (p_sev >= mass_thr)
    if mask.any():
        out[mask] = 3 + probs[mask][:, 3:5].argmax(axis=1)
    return out


def abstain_mask(
    probs: np.ndarray,
    preds: np.ndarray,
    conf_thr: float,
    early_only: bool = True,
) -> np.ndarray:
    """Abstain on low peak-prob; by default only when predicted grade ≤2."""
    conf = probs.max(axis=1)
    ab = conf < conf_thr
    if early_only:
        ab = ab & (preds <= 2)
    # never abstain after severe upgrade targets (preds 3/4)
    ab = ab & (preds <= 2)
    return ab.astype(bool)


def covered_kl_recall(y: np.ndarray, pred: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    if keep.sum() == 0:
        return {f"kl{g}": 0.0 for g in range(5)}
    pc = per_class_accuracy(y[keep], pred[keep])
    return {f"kl{g}": float(pc[g]) for g in range(5)}


def passes_full_gate(m: dict, prefix="test_") -> bool:
    return (
        m[f"{prefix}qwk"] + 1e-12 >= FLOOR_QWK
        and m[f"{prefix}acc_kl1"] + 1e-12 >= FLOOR_KL1
        and m[f"{prefix}acc_kl2"] + 1e-12 >= FLOOR_KL2
        and m[f"{prefix}acc_kl3"] + 1e-12 >= FLOOR_KL3
        and m[f"{prefix}acc_kl4"] + 1e-12 >= FLOOR_KL4
    )


def eval_policy(
    y: np.ndarray,
    bin_l: torch.Tensor,
    oa_l: torch.Tensor,
    T: float,
    sev_thr: float,
    conf_thr: float,
    prefix: str,
) -> dict:
    probs = calibrated_probs(bin_l, oa_l, T).numpy()
    raw = s6_predict_with_threshold(bin_l / max(T, 1e-3), oa_l / max(T, 1e-3), OA_THRESH).numpy()
    # argmax of calibrated probs is usually close; keep cascade decode for stability
    preds = apply_severe_protect(probs, raw, sev_thr)
    ab = abstain_mask(probs, preds, conf_thr, early_only=True)

    full = compute_all_metrics(y, preds, prefix=prefix)
    ab_m = metrics_with_abstain(y, preds, ab)
    keep = ~ab
    kl_cov = covered_kl_recall(y, preds, keep)

    out = {**full}
    out.update({f"{prefix}{k}": v for k, v in ab_m.items()})
    out.update({f"{prefix}covered_acc_kl{g}": kl_cov[f"kl{g}"] for g in range(5)})
    out[f"{prefix}severe_protect_thr"] = sev_thr
    out[f"{prefix}abstain_conf_thr"] = conf_thr
    out[f"{prefix}temperature"] = T
    out["_preds"] = preds
    out["_abstain"] = ab
    out["_probs"] = probs
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    print("\n" + "=" * 60)
    print("  STAGE 10 — CLINICAL DEPLOY (S6 + temp + abstain + severe-protect)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Min coverage: {MIN_COVERAGE}")
    print(
        f"  Promote gate (full, no abstain): QWK>={FLOOR_QWK} "
        f"KL1>={FLOOR_KL1} KL2>={FLOOR_KL2} KL3>={FLOOR_KL3} KL4>={FLOOR_KL4}"
    )

    if not S6_CKPT.exists():
        print(f"[FAIL] Missing {S6_CKPT}")
        sys.exit(1)

    loaders = build_dataloaders(
        root_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
        channels=3,
        mean=mean,
        std=std,
        use_weighted_sampler=False,
        pin_memory=device.type == "cuda",
    )

    model = DualStreamCascade(pretrained=False, dropout=0.5, local_frac=0.72).to(device)
    ckpt = torch.load(S6_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded {S6_CKPT.name} (epoch={ckpt.get('epoch')}, val_qwk={ckpt.get('val_qwk')})")

    print("\n[1/4] Cache val/test logits...")
    y_va, b_va, o_va = collect_logits(model, loaders["val"], device)
    y_te, b_te, o_te = collect_logits(model, loaders["test"], device)

    # ── Champion card (raw S6, T=1, no protect, no abstain) ─────────────────
    print("\n[2/4] Champion card (raw S6)...")
    raw_te = s6_predict_with_threshold(b_te, o_te, OA_THRESH).numpy()
    champ_m = compute_all_metrics(y_te, raw_te, prefix="test_")
    print_metrics(champ_m, title="Champion card — raw S6 (full test)")

    print("\n[3/4] Fit temperature on VAL...")
    T = fit_temperature(b_va, o_va, y_va)
    print(f"  Temperature T = {T:.4f}")

    # Grid: severe mass threshold + abstain confidence
    sev_grid = [1.0, 0.45, 0.40, 0.35, 0.30, 0.25]  # 1.0 = off
    conf_grid = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65]  # 0.0 = never abstain

    print("\n[4/4] Sweep abstain + severe-protect on VAL...")
    best = None
    n_ok = 0
    for sev_thr, conf_thr in itertools.product(sev_grid, conf_grid):
        m = eval_policy(y_va, b_va, o_va, T, sev_thr, conf_thr, prefix="val_")
        cov = m["val_coverage"]
        if cov + 1e-12 < MIN_COVERAGE:
            continue
        n_ok += 1
        # Prefer high covered QWK; protect covered KL3/4; prefer higher coverage on ties
        score = (
            m["val_qwk_covered"],
            m["val_covered_acc_kl3"],
            m["val_covered_acc_kl4"],
            cov,
        )
        cand = {
            "sev_thr": sev_thr,
            "conf_thr": conf_thr,
            "score": score,
            "val": {k: v for k, v in m.items() if not k.startswith("_")},
        }
        if best is None or score > best["score"]:
            best = cand

    if best is None:
        print("  [warn] No config met min coverage; relaxing to max covered QWK")
        for sev_thr, conf_thr in itertools.product(sev_grid, conf_grid):
            m = eval_policy(y_va, b_va, o_va, T, sev_thr, conf_thr, prefix="val_")
            score = (m["val_qwk_covered"], m["val_coverage"])
            cand = {
                "sev_thr": sev_thr,
                "conf_thr": conf_thr,
                "score": score,
                "val": {k: v for k, v in m.items() if not k.startswith("_")},
            }
            if best is None or score > best["score"]:
                best = cand
    else:
        print(f"  Feasible configs (coverage>={MIN_COVERAGE}): {n_ok}")

    print(
        f"  Best VAL: sev_thr={best['sev_thr']} conf_thr={best['conf_thr']}  "
        f"cov={best['val']['val_coverage']:.3f}  "
        f"qwk_cov={best['val']['val_qwk_covered']:.4f}  "
        f"KL3_cov={best['val']['val_covered_acc_kl3']:.3f}  "
        f"KL4_cov={best['val']['val_covered_acc_kl4']:.3f}"
    )

    # Apply on test
    te = eval_policy(
        y_te, b_te, o_te, T, best["sev_thr"], best["conf_thr"], prefix="test_"
    )
    clinical_full = {k: v for k, v in te.items() if not k.startswith("_")}
    print_metrics(
        {k: v for k, v in clinical_full.items() if k.startswith("test_") and "covered" not in k and "abstain" not in k and "thr" not in k and "temperature" not in k and "n_" not in k},
        title="Clinical card — full test (after severe-protect, before abstain filter)",
    )
    print(
        f"\n  Clinical abstain: coverage={te['test_coverage']:.3f}  "
        f"qwk_covered={te['test_qwk_covered']:.4f}  "
        f"acc_covered={te['test_acc_covered']:.4f}  "
        f"n_abstain={int(te['test_n_abstain'])}"
    )
    print(
        f"  Covered KL recalls: "
        + " ".join(f"{g}={te[f'test_covered_acc_kl{g}']:.3f}" for g in range(5))
    )

    # Full-test gate on severe-protect only (conf_thr=0 → no abstain)
    te_full_prot = eval_policy(y_te, b_te, o_te, T, best["sev_thr"], 0.0, prefix="test_")
    full_prot_m = {
        k: v for k, v in te_full_prot.items()
        if k.startswith("test_") and not k.startswith("_")
        and "covered" not in k and k not in ("test_n_abstain", "test_n_covered", "test_coverage", "test_qwk_covered", "test_acc_covered")
        and "thr" not in k and "temperature" not in k and "covered_acc" not in k
    }
    # Recompute clean full metrics keys only
    full_prot_clean = compute_all_metrics(y_te, te_full_prot["_preds"], prefix="test_")
    gate_ok = passes_full_gate(full_prot_clean)
    print(
        f"\n  Full-test severe-protect (no abstain): "
        f"QWK={full_prot_clean['test_qwk']:.4f} "
        f"KL1={full_prot_clean['test_acc_kl1']:.3f} "
        f"KL2={full_prot_clean['test_acc_kl2']:.3f} "
        f"KL3={full_prot_clean['test_acc_kl3']:.3f} "
        f"KL4={full_prot_clean['test_acc_kl4']:.3f}"
    )
    print(f"  Promote gate: {'PASS' if gate_ok else 'REJECT — champion JSON untouched'}")

    promoted = False
    if gate_ok and full_prot_clean["test_qwk"] + 1e-12 >= champ_m["test_qwk"]:
        champ = {
            "status": "champion",
            "stage": "10",
            "reason": "S6 + temperature + severe-protect cleared full-test gate",
            "test_qwk": round(full_prot_clean["test_qwk"], 4),
            "checkpoint": "outputs/models/resnet50_dual_cascade_best.pth",
            "params": {
                "temperature": round(T, 4),
                "severe_protect_thr": best["sev_thr"],
                "abstain_conf_thr": best["conf_thr"],
            },
            "previous_champion_qwk": FLOOR_QWK,
        }
        CHAMPION_JSON.write_text(json.dumps(champ, indent=2), encoding="utf-8")
        promoted = True
        print("  ✅ Champion JSON updated → Stage 10")
    else:
        print("  ✅ Champion JSON unchanged (Stage 6).")

    def round_dict(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, (float, np.floating)):
                out[k] = round(float(v), 4)
            elif isinstance(v, (int, np.integer)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    report = {
        "stage": "10",
        "technique": "frozen S6 + temperature + severe-protect + abstain",
        "checkpoint": str(S6_CKPT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "params": {
            "temperature": round(T, 4),
            "severe_protect_thr": best["sev_thr"],
            "abstain_conf_thr": best["conf_thr"],
            "min_coverage": MIN_COVERAGE,
            "oa_threshold": OA_THRESH,
        },
        "cards": {
            "champion": {
                "description": "Raw Stage 6 predictions on all test cases (weights frozen)",
                "metrics": round_dict(champ_m),
            },
            "clinical": {
                "description": (
                    "Calibrated probs + severe-protect + abstain on early low-confidence; "
                    "use covered_* when the system answers"
                ),
                "val_selection": round_dict(best["val"]),
                "test_full_after_severe_protect": round_dict(full_prot_clean),
                "test_with_abstain": round_dict(
                    {k: v for k, v in clinical_full.items() if not k.startswith("_")}
                ),
            },
        },
        "gate": {
            "floor_qwk": FLOOR_QWK,
            "floor_kl1": FLOOR_KL1,
            "floor_kl2": FLOOR_KL2,
            "floor_kl3": FLOOR_KL3,
            "floor_kl4": FLOOR_KL4,
            "evaluated_on": "full test with severe-protect, no abstain",
            "passed": gate_ok,
        },
        "promoted": promoted,
        "champion_json_untouched": not promoted,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Saved → {OUT_JSON}")
    print(
        f"\n  Summary: champion QWK={champ_m['test_qwk']:.4f} | "
        f"clinical covered QWK={te['test_qwk_covered']:.4f} "
        f"(coverage={te['test_coverage']:.1%})"
    )


if __name__ == "__main__":
    main()
