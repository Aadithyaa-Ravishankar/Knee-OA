"""
stage3_phase01_champion.py
===========================
Phase 0 + Phase 1 (validation only, no training, test locked).

Phase 0: bootstrap 95% CIs for the Stage 6 champion and past candidates
         (Stage 9b, 16, 18 Arm A), plus paired bootstrap differences vs champion.
Phase 1: inference-only decode options on the champion:
         - horizontal-flip TTA
         - rank-consistency rate of the OA thresholds
         - prior correction for the KL1-heavy Stage 6 sampler
         - expected grade + QWK cutpoints (cross-fit by patient group, so not fit on the scored images)

Usage:
    python src/training/stage3_phase01_champion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.decoding import (
    crossfit_cutpoints,
    expected_grade,
    prior_correct,
    rank_inconsistency_rate,
)
from src.evaluation.ensemble import s6_class_probs, s6_predict_with_threshold
from src.evaluation.protocol import bootstrap_ci, metric_bundle, paired_bootstrap_diff, patient_id
from src.models.dual_stream_cascade import DualStreamCascade
from src.preprocessing.dataset import KneeOADataset
from src.preprocessing.transforms import get_eval_transforms_3ch

DATA = PROJECT_ROOT / "Dataset_combined"
MODELS = PROJECT_ROOT / "outputs" / "models"
OUT = PROJECT_ROOT / "reports" / "pipeline"
MEAN = [0.578707] * 3
STD = [0.145419] * 3

CANDIDATES = {
    "stage6_champion": MODELS / "resnet50_dual_cascade_best.pth",
    "stage9b_sampler_ft": MODELS / "resnet50_s6_sampler_ft_best.pth",
    "stage16_t0x2": MODELS / "stage16_kl12_candidate.pth",
    "stage18_armA": MODELS / "stage18_armA_candidate.pth",
}
STAGE6_SCALE = {0: 0.45, 1: 1.35, 2: 0.75, 3: 1.0, 4: 1.15}


@torch.no_grad()
def infer(model, loader, dev):
    bins, oas, bins_f, oas_f, ys = [], [], [], [], []
    for x, y in tqdm(loader, desc="  infer", leave=False, ncols=80):
        x = x.to(dev)
        b, o = model(x)
        bf, of = model(torch.flip(x, dims=[3]))
        bins.append(b.cpu()); oas.append(o.cpu()); bins_f.append(bf.cpu()); oas_f.append(of.cpu())
        ys.append(y)
    cat = lambda t: torch.cat(t)
    return cat(bins), cat(oas), cat(bins_f), cat(oas_f), cat(ys).numpy()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = KneeOADataset(DATA, "val", transform=get_eval_transforms_3ch(MEAN, STD))
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    groups_patient = np.array([patient_id(p.stem) for p, _ in ds.samples])
    # 5 patient-disjoint groups for cross-fitting cutpoints
    uniq = np.unique(groups_patient)
    rng = np.random.default_rng(0)
    gmap = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
    groups = np.array([gmap[g] for g in groups_patient])

    results, preds = {}, {}
    y = None
    for name, ckpt in CANDIDATES.items():
        if not ckpt.exists():
            print(f"[skip] {name}: missing {ckpt.name}")
            continue
        model = DualStreamCascade(pretrained=False, dropout=0.5, local_frac=0.72).to(dev)
        try:
            model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model_state_dict"])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: incompatible checkpoint ({e.__class__.__name__})")
            continue
        model.eval()
        print(f"[{name}] inference on val (n={len(ds)}) ...")
        b, o, bf, of, y = infer(model, loader, dev)
        pred = s6_predict_with_threshold(b, o, 0.5).numpy()
        pred_tta = s6_predict_with_threshold((b + bf) / 2, (o + of) / 2, 0.5).numpy()
        preds[name] = pred
        preds[f"{name}+flipTTA"] = pred_tta
        results[name] = {"metrics": metric_bundle(y, pred), "ci": bootstrap_ci(y, pred)}
        results[f"{name}+flipTTA"] = {"metrics": metric_bundle(y, pred_tta), "ci": bootstrap_ci(y, pred_tta)}

        if name == "stage6_champion":
            probs = s6_class_probs((b + bf) / 2, (o + of) / 2).numpy()
            thr = torch.sigmoid((o + of) / 2).numpy()
            train_counts = KneeOADataset(DATA, "train").class_counts()
            chen_counts = {k: sum(1 for p, l in KneeOADataset(DATA, "train").samples
                                  if l == k and not p.name.startswith("gp_")) for k in range(5)}
            natural = np.array([chen_counts[k] for k in range(5)], dtype=float)
            natural /= natural.sum()
            sampler = np.array([STAGE6_SCALE[k] for k in range(5)], dtype=float)
            sampler /= sampler.sum()
            p_corr = prior_correct(probs, sampler, natural)

            variants = {
                "champion+TTA+argmax": probs.argmax(1),
                "champion+TTA+prior_correct_argmax": p_corr.argmax(1),
                "champion+TTA+expected_grade_cutpoints(crossfit)": crossfit_cutpoints(expected_grade(probs), y, groups),
                "champion+TTA+prior_correct+cutpoints(crossfit)": crossfit_cutpoints(expected_grade(p_corr), y, groups),
            }
            for vname, vp in variants.items():
                preds[vname] = vp
                results[vname] = {"metrics": metric_bundle(y, vp), "ci": bootstrap_ci(y, vp)}
            results["_champion_diagnostics"] = {
                "oa_threshold_rank_inconsistency_rate": rank_inconsistency_rate(thr),
                "sampler_prior_stage6": sampler.round(4).tolist(),
                "natural_prior_chen_train": natural.round(4).tolist(),
                "train_counts_incl_gp": train_counts,
            }
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    base = preds["stage6_champion"]
    paired = {k: paired_bootstrap_diff(y, base, v) for k, v in preds.items() if k != "stage6_champion"}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase01_summary.json").write_text(
        json.dumps({"split": "val", "n": int(len(y)), "results": results, "paired_vs_champion": paired}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Pipeline Phase 0/1 — champion and candidates on validation (bootstrap 95% CI)",
        "",
        f"n = {len(y)} validation images. Test split not loaded.",
        "",
        "| Model / decode | QWK [95% CI] | Macro-F1 | Acc | KL0→KL1 | KL2→KL1 | Pred KL1 / true KL1 | ΔQWK vs champion [CI] | Sig? |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for k, r in results.items():
        if k.startswith("_"):
            continue
        m, ci = r["metrics"], r["ci"]["qwk"]
        if k == "stage6_champion":
            d = "—"
            sig = "—"
        else:
            p = paired[k]["qwk"]
            d = f"{p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
            sig = "yes" if p["significant"] else "no"
        lines.append(
            f"| {k} | {m['qwk']:.4f} [{ci['lo']:.3f}, {ci['hi']:.3f}] | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | "
            f"{m['kl0_to_kl1']} | {m['kl2_to_kl1']} | {m['pred_kl1_count']} / {m['true_kl1_count']} | {d} | {sig} |"
        )
    diag = results.get("_champion_diagnostics", {})
    lines += [
        "",
        f"- Champion OA-threshold rank-inconsistency rate: **{diag.get('oa_threshold_rank_inconsistency_rate', float('nan')):.4f}**",
        f"- Stage 6 sampler prior: {diag.get('sampler_prior_stage6')}  vs natural Chen prior: {diag.get('natural_prior_chen_train')}",
        "- Cutpoint variants are cross-fit: cutpoints for each patient group are fit on the other groups.",
    ]
    (OUT / "phase01_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
