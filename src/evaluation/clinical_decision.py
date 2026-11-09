"""
clinical_decision.py
====================
Safe healthcare-oriented decision layer on Stage 6 cascade probs.

Outputs (never invent certainty):
  KL-0, KL-1, KL-2, EARLY_OA_1_2, KL-3, KL-4, REFER

IMPORTANT
---------
No ML system can guarantee zero misclassifications. This layer *reduces risk*
by (1) collapsing uncertain 1↔2 into EARLY_OA_1_2, (2) REFER when confidence
is too low for autonomous use, and (3) attaching explanations for review.
Final clinical responsibility remains with a qualified clinician.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


LABELS = ("KL-0", "KL-1", "KL-2", "EARLY_OA_1_2", "KL-3", "KL-4", "REFER")


@dataclass
class ClinicalDecision:
    label: str
    kl_exact: int | None          # 0..4 if exact grade; None for band/refer
    confidence: float
    probs: dict[str, float]       # KL-0..4 calibrated probs
    reasons: list[str]            # human-readable why
    needs_review: bool
    explain_focus: str            # what Grad-CAM / UI should highlight


def _round_probs(p: np.ndarray) -> dict[str, float]:
    return {f"KL-{i}": round(float(p[i]), 4) for i in range(5)}


def decide_one(
    probs: np.ndarray,
    *,
    exact_conf: float = 0.55,
    early_band_conf: float = 0.45,
    early_margin: float = 0.12,
    severe_mass_thr: float = 0.45,
    normal_thr: float = 0.55,
    refer_conf: float = 0.40,
) -> ClinicalDecision:
    """
    probs : (5,) calibrated P(KL=k)

    Policy (high level)
    -------------------
    1) Strong severe mass → KL-3 / KL-4 (protect severe pathway)
    2) Strong normal → KL-0
    3) Early OA mass dominant:
         - clear 1 vs 2 margin + high conf → exact KL-1 or KL-2
         - otherwise → EARLY_OA_1_2 (uncertain mid-grade, not a forced pick)
    4) Else → REFER (do not auto-grade)
    """
    p = np.asarray(probs, dtype=np.float64).reshape(5)
    p = p / max(p.sum(), 1e-8)
    conf = float(p.max())
    arg = int(p.argmax())
    p_sev = float(p[3] + p[4])
    p_early = float(p[1] + p[2])
    p0 = float(p[0])
    margin_12 = abs(float(p[1] - p[2]))

    # ── Severe protect ───────────────────────────────────────────────────
    if p_sev >= severe_mass_thr and conf >= refer_conf:
        grade = 3 + int(p[3:5].argmax())
        peak = float(p[grade])
        # Precision mode: only emit exact KL-3/4 if the peak itself clears exact_conf
        if peak < exact_conf:
            reasons = [
                f"Severe mass P(KL-3)+P(KL-4)={p_sev:.2f} ≥ {severe_mass_thr:.2f}, "
                f"but peak P(KL-{grade})={peak:.2f} < {exact_conf:.2f}.",
                "REFER — refusing low-precision severe call.",
            ]
            return ClinicalDecision(
                label="REFER",
                kl_exact=None,
                confidence=peak,
                probs=_round_probs(p),
                reasons=reasons,
                needs_review=True,
                explain_focus="severe_mass_without_sharp_peak",
            )
        reasons = [
            f"Combined P(KL-3)+P(KL-4)={p_sev:.2f} ≥ {severe_mass_thr:.2f} (severe-protect).",
            f"Within severe grades, P(KL-{grade})={peak:.2f} ≥ {exact_conf:.2f}.",
            "Joint-space loss / advanced OA cues dominate the cascade probability mass.",
        ]
        return ClinicalDecision(
            label=f"KL-{grade}",
            kl_exact=grade,
            confidence=peak,
            probs=_round_probs(p),
            reasons=reasons,
            needs_review=False,
            explain_focus="severe_joint_space_and_bone_remodeling",
        )

    # ── Normal ───────────────────────────────────────────────────────────
    if p0 >= normal_thr and p0 >= p_early and p0 >= p_sev:
        reasons = [
            f"P(KL-0)={p0:.2f} ≥ {normal_thr:.2f} and exceeds early/severe mass.",
            "Cascade binary head favors non-OA; ordinal mass not concentrated on 1–4.",
        ]
        needs = p0 < exact_conf
        if needs:
            reasons.append("Confidence below exact-grade threshold → flag for light review.")
        return ClinicalDecision(
            label="KL-0",
            kl_exact=0,
            confidence=p0,
            probs=_round_probs(p),
            reasons=reasons,
            needs_review=needs,
            explain_focus="global_joint_appearance_no_clear_oa",
        )

    # ── Conflict normal vs early → REFER (do not guess) ──────────────────
    if min(p0, p_early) >= 0.30 and abs(p0 - p_early) < 0.15 and p_sev < 0.25:
        reasons = [
            f"Conflicting mass between normal and early OA (P0={p0:.2f}, Pearly={p_early:.2f}).",
            "Healthcare-safe action: REFER rather than risk KL-0 ↔ early-OA error.",
        ]
        return ClinicalDecision(
            label="REFER",
            kl_exact=None,
            confidence=conf,
            probs=_round_probs(p),
            reasons=reasons,
            needs_review=True,
            explain_focus="normal_vs_early_oa_conflict",
        )

    # ── Early OA band (1–2) ──────────────────────────────────────────────
    # Require clear early dominance over normal (avoid labeling true KL-0 as early)
    early_clear = (
        p_early >= early_band_conf
        and p_early >= p0 + 0.18
        and p_early >= p_sev + 0.05
    )
    if early_clear:
        # Exact 1 or 2 only if margin is clear AND peak is strong
        if margin_12 >= early_margin and conf >= exact_conf and arg in (1, 2):
            reasons = [
                f"Early-OA mass P(1)+P(2)={p_early:.2f} dominates (vs P0={p0:.2f}).",
                f"|P(KL-1)-P(KL-2)|={margin_12:.2f} ≥ {early_margin:.2f} → grades separable.",
                f"Exact grade KL-{arg} with peak probability {p[arg]:.2f}.",
            ]
            return ClinicalDecision(
                label=f"KL-{arg}",
                kl_exact=arg,
                confidence=float(p[arg]),
                probs=_round_probs(p),
                reasons=reasons,
                needs_review=False,
                explain_focus="mild_osteophyte_or_doubtful_narrowing",
            )

        reasons = [
            f"Early-OA mass P(1)+P(2)={p_early:.2f} dominates, but 1↔2 is ambiguous.",
            f"|P(KL-1)-P(KL-2)|={margin_12:.2f} < {early_margin:.2f} or peak conf {conf:.2f} < {exact_conf:.2f}.",
            "Healthcare-safe action: report EARLY_OA_1_2 instead of forcing KL-1 vs KL-2.",
            "Explanation: model sees OA-leaning cues but cannot reliably separate doubtful vs mild.",
            "Clinician should adjudicate osteophytes / joint-space narrowing.",
        ]
        return ClinicalDecision(
            label="EARLY_OA_1_2",
            kl_exact=None,
            confidence=p_early,
            probs=_round_probs(p),
            reasons=reasons,
            needs_review=True,
            explain_focus="uncertain_early_oa_osteophyte_jsn_boundary",
        )

    # ── Fallback exact only for non-mid grades with very high conf ───────
    if conf >= exact_conf and arg in (0, 3, 4):
        reasons = [
            f"Peak class KL-{arg} with confidence {conf:.2f} ≥ {exact_conf:.2f}.",
            "Exact grade issued; probabilities + Grad-CAM still shown for audit.",
        ]
        return ClinicalDecision(
            label=f"KL-{arg}",
            kl_exact=arg,
            confidence=conf,
            probs=_round_probs(p),
            reasons=reasons,
            needs_review=False,
            explain_focus="top_class_evidence",
        )

    # Mid-grade argmax without clear band → REFER (never force 1/2)
    if arg in (1, 2):
        reasons = [
            f"Model leans KL-{arg} (p={p[arg]:.2f}) but fails exact/band safety checks.",
            f"P0={p0:.2f}, Pearly={p_early:.2f}, margin_12={margin_12:.2f}.",
            "REFER — refusing autonomous KL-1/KL-2 label under ambiguity.",
        ]
        return ClinicalDecision(
            label="REFER",
            kl_exact=None,
            confidence=conf,
            probs=_round_probs(p),
            reasons=reasons,
            needs_review=True,
            explain_focus="unsafe_mid_grade_guess",
        )

    # ── REFER ────────────────────────────────────────────────────────────
    reasons = [
        f"No safe autonomous decision: peak conf={conf:.2f}, P0={p0:.2f}, "
        f"Pearly={p_early:.2f}, Psevere={p_sev:.2f}.",
        "Below refer thresholds for normal / early-band / severe-protect.",
        "REFER to clinician — model withholds a grade rather than risk a wrong label.",
    ]
    return ClinicalDecision(
        label="REFER",
        kl_exact=None,
        confidence=conf,
        probs=_round_probs(p),
        reasons=reasons,
        needs_review=True,
        explain_focus="low_confidence_or_conflicting_mass",
    )


def decide_batch(probs: np.ndarray, **kwargs) -> list[ClinicalDecision]:
    probs = np.asarray(probs)
    return [decide_one(probs[i], **kwargs) for i in range(len(probs))]


def is_band_correct(y_true: int, decision: ClinicalDecision) -> bool | None:
    """
    Scoring for selective prediction.
    Returns True/False if model issued a grade/band; None if REFER (no error credited).
    """
    if decision.label == "REFER":
        return None
    if decision.label == "EARLY_OA_1_2":
        return y_true in (1, 2)
    if decision.kl_exact is None:
        return None
    return int(decision.kl_exact) == int(y_true)


def decision_to_dict(d: ClinicalDecision) -> dict[str, Any]:
    return asdict(d)
