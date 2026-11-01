"""
ensemble.py
===========
Safe Stage 5 + Stage 6 ensembling helpers.

S5: DualStream CORN → 5-class probs via corn_probs
S6: Cascade → P(0)=1-p_oa; P(1..4)=p_oa * OA_CORN_probs
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.ordinal_loss import corn_probs, corn_predict


def s5_class_probs(logits_corn4: torch.Tensor) -> torch.Tensor:
    """(B, 4) CORN logits → (B, 5) class probs."""
    return corn_probs(logits_corn4)


def s6_class_probs(
    bin_logit: torch.Tensor,
    oa_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Cascade heads → (B, 5) probs.
    bin_logit : (B, 1)  OA logit
    oa_logits : (B, 3)  CORN for remapped grades 0..3 (= KL 1..4)
    """
    p_oa = torch.sigmoid(bin_logit.squeeze(1))  # (B,)
    p_not = 1.0 - p_oa
    oa_p = corn_probs(oa_logits)  # (B, 4) over KL1..4
    # (B, 5)
    probs = torch.zeros(bin_logit.size(0), 5, device=bin_logit.device, dtype=oa_p.dtype)
    probs[:, 0] = p_not
    probs[:, 1:] = p_oa.unsqueeze(1) * oa_p
    # renorm for safety
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


def s6_predict_with_threshold(
    bin_logit: torch.Tensor,
    oa_logits: torch.Tensor,
    oa_threshold: float = 0.5,
) -> torch.Tensor:
    p_oa = torch.sigmoid(bin_logit.squeeze(1))
    is_oa = p_oa >= oa_threshold
    grades = torch.zeros(bin_logit.size(0), dtype=torch.long, device=bin_logit.device)
    if is_oa.any():
        grades[is_oa] = corn_predict(oa_logits[is_oa]) + 1
    return grades


def blend_probs(p_a: torch.Tensor, p_b: torch.Tensor, w_a: float = 0.5) -> torch.Tensor:
    w_a = float(w_a)
    out = w_a * p_a + (1.0 - w_a) * p_b
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)


def probs_to_pred(probs: torch.Tensor) -> torch.Tensor:
    return probs.argmax(dim=1)
