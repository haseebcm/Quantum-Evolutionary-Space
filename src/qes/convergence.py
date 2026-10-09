"""Convergence metrics (docs/QES-architecture.md, section 24).

    H_Q      = - sum_i p_i * ln(p_i)
    H_max    = ln(N)
    H_bar_Q  = H_Q / H_max                  (normalized uncertainty)
    C_Q      = 1 - H_bar_Q                  (convergence coefficient)

C_Q measures normalized weight concentration, not geometric or objective
convergence. Raw nonnegative scores are normalized before computing entropy.
Empty and all-zero populations have no evidence of concentration (C_Q = 0).

Additional diversity/convergence diagnostics:

    KL(p || q) = sum_i p_i * ln(p_i / q_i)   (relative entropy vs. a reference)
    Gini(p)                                  (inequality of the weight distribution)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def qes_entropy(weights: Sequence[float]) -> float:
    """H_Q = - sum_i p_i * ln(p_i), with the convention 0*ln(0) = 0."""
    p = _normalize(weights)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(weights: Sequence[float]) -> float:
    """H_bar_Q = H_Q / H_max, H_max = ln(N). Returns 0 when N <= 1."""
    p = _normalize(weights)
    n = len(p)
    if n <= 1:
        return 0.0
    h_max = np.log(n)
    return float(np.clip(qes_entropy(p.tolist()) / h_max, 0.0, 1.0))


def convergence_coefficient(weights: Sequence[float]) -> float:
    """C_Q = 1 - H_bar_Q."""
    p = _normalize(weights)
    if p.size == 0 or not np.any(p):
        return 0.0
    return 1.0 - normalized_entropy(p.tolist())


def _normalize(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be a finite, nonnegative one-dimensional sequence")
    # Scale first to avoid overflow when several large finite scores are added.
    maximum = float(np.max(w)) if w.size else 0.0
    if maximum == 0.0:
        return w.copy()
    scaled = w / maximum
    return scaled / scaled.sum()


def kl_divergence(weights: Sequence[float], reference: Sequence[float]) -> float:
    """KL(p || q) = sum_i p_i * ln(p_i / q_i), p/q renormalized to sum to 1.

    Measures how far the current room-weight distribution `weights` has
    diverged from a `reference` distribution (e.g. the uniform prior, or an
    earlier tick's weights) -- a directional alternative to the symmetric
    entropy-based convergence coefficient, useful for detecting drift.
    """
    p = _normalize(weights)
    q = _normalize(reference)
    if p.shape != q.shape:
        raise ValueError("weights and reference must have the same length")
    mask = p > 0
    # 0 * ln(0/q) = 0 by convention; q_i = 0 where p_i > 0 gives +inf (true KL semantics).
    q_safe = np.where(q[mask] > 0, q[mask], np.finfo(float).tiny)
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe)))


def gini_coefficient(weights: Sequence[float]) -> float:
    """Gini coefficient of the weight distribution in [0, 1].

    0 = perfectly uniform (every room equally likely); 1 = fully
    concentrated on a single room. A complementary lens to the
    entropy-based convergence coefficient that is more sensitive to the
    shape of the tail of the distribution.
    """
    w = np.sort(_normalize(weights))
    n = w.shape[0]
    if n == 0 or w.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * w)) / (n * w.sum()) - (n + 1) / n)
