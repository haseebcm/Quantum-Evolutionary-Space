"""Convergence metrics (docs/QES-architecture.md, section 24).

    H_Q      = - sum_i p_i * ln(p_i)
    H_max    = ln(N)
    H_bar_Q  = H_Q / H_max                  (normalized uncertainty)
    C_Q      = 1 - H_bar_Q                  (convergence coefficient)

C_Q -> 1 means QES realities strongly converge; C_Q -> 0 means the
possibility space remains dispersed.

Additional diversity/convergence diagnostics:

    KL(p || q) = sum_i p_i * ln(p_i / q_i)   (relative entropy vs. a reference)
    Gini(p)                                  (inequality of the weight distribution)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def qes_entropy(weights: Sequence[float]) -> float:
    """H_Q = - sum_i p_i * ln(p_i), with the convention 0*ln(0) = 0."""
    p = np.asarray(weights, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(weights: Sequence[float]) -> float:
    """H_bar_Q = H_Q / H_max, H_max = ln(N). Returns 0 when N <= 1."""
    n = len(weights)
    if n <= 1:
        return 0.0
    h_max = np.log(n)
    return qes_entropy(weights) / h_max


def convergence_coefficient(weights: Sequence[float]) -> float:
    """C_Q = 1 - H_bar_Q."""
    return 1.0 - normalized_entropy(weights)


def _normalize(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    return w / total if total > 0 else w


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
    w = np.sort(np.asarray(weights, dtype=float))
    n = w.shape[0]
    if n == 0 or w.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * w)) / (n * w.sum()) - (n + 1) / n)
