"""
decoding.py
===========
Turn 5-class KL probabilities into hard grades.

- argmax
- expected grade + cutpoints optimized for QWK
- prior (logit) correction for a training sampler that distorts class frequencies
- rank-consistency check for cumulative threshold heads
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score

DEFAULT_CUTS = np.array([0.5, 1.5, 2.5, 3.5])


def expected_grade(probs: np.ndarray) -> np.ndarray:
    return probs @ np.arange(probs.shape[1], dtype=np.float64)


def apply_cutpoints(scores: np.ndarray, cuts) -> np.ndarray:
    return np.searchsorted(np.sort(np.asarray(cuts, dtype=np.float64)), scores, side="right")


def _qwk(y, p) -> float:
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def fit_cutpoints(scores: np.ndarray, y: np.ndarray, n_rounds: int = 3, grid: int = 41) -> np.ndarray:
    """Coordinate search of 4 cutpoints maximizing QWK; starts at 0.5/1.5/2.5/3.5."""
    cuts = DEFAULT_CUTS.copy()
    best = _qwk(y, apply_cutpoints(scores, cuts))
    for _ in range(n_rounds):
        for i in range(len(cuts)):
            lo = cuts[i - 1] + 0.05 if i > 0 else 0.0
            hi = cuts[i + 1] - 0.05 if i < len(cuts) - 1 else 4.0
            for c in np.linspace(lo, hi, grid):
                trial = cuts.copy()
                trial[i] = c
                q = _qwk(y, apply_cutpoints(scores, trial))
                if q > best + 1e-9:
                    best, cuts = q, trial
    return cuts


def crossfit_cutpoints(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-group predictions: cutpoints for group g are fit on all other groups."""
    pred = np.zeros(len(y), dtype=np.int64)
    for g in np.unique(groups):
        m = groups == g
        cuts = fit_cutpoints(scores[~m], y[~m])
        pred[m] = apply_cutpoints(scores[m], cuts)
    return pred


def prior_correct(probs: np.ndarray, train_prior: np.ndarray, target_prior: np.ndarray) -> np.ndarray:
    """Re-weight p(y|x) from the sampler-induced prior to the true prior (Bayes rule)."""
    ratio = np.asarray(target_prior, dtype=np.float64) / np.clip(np.asarray(train_prior, dtype=np.float64), 1e-8, None)
    out = probs * ratio[None, :]
    return out / out.sum(axis=1, keepdims=True)


def sampler_prior(class_counts: dict, class_scale: dict | None) -> np.ndarray:
    """Expected class frequency under inverse-frequency sampling x class_scale."""
    scale = class_scale or {}
    mass = np.array([float(scale.get(k, 1.0)) for k in range(5)])
    if not any(class_counts.values()):
        return np.full(5, 0.2)
    return mass / mass.sum()


def rank_inconsistency_rate(threshold_probs: np.ndarray) -> float:
    """Fraction of samples where P(y>k+1) > P(y>k) for some k (non-monotone thresholds)."""
    if threshold_probs.shape[1] < 2:
        return 0.0
    return float((np.diff(threshold_probs, axis=1) > 1e-6).any(axis=1).mean())
