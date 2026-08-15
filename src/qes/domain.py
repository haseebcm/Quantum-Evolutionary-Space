"""Domain Nullification (docs/QES-architecture.md, section 4).

A 33-domain activation vector a = [a_1, ..., a_33] determines what exists
inside a room:

    a_k = 1        -> domain k is active
    a_k = 0        -> domain k is nullified
    0 < a_k < 1    -> domain k is partially constrained

    A_i = diag(a_i)
    x~_i = A_i * x_i                              (effective state)

    W(a) = sum_{k=1}^{33} a_k * W^(k)              (Domain-Nullified Metric Constructor)
    W_i = W(a_i)

This lets every QES room operate under a different active domain geometry:
R_i != R_j not merely because their states differ, but because their active
dimensional structures can differ.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NUM_DOMAINS = 33


class DomainNullification:
    """Builds the domain-nullified metric W(a) from per-domain metrics W^(k)."""

    def __init__(self, domain_metrics: Sequence[np.ndarray]):
        """domain_metrics: sequence of W^(k) matrices, one per domain (len <= 33)."""
        self.domain_metrics = [np.asarray(w, dtype=float) for w in domain_metrics]
        if len(self.domain_metrics) > NUM_DOMAINS:
            raise ValueError(f"at most {NUM_DOMAINS} domain metrics are supported")

    @staticmethod
    def activation_matrix(a: np.ndarray) -> np.ndarray:
        """A = diag(a)."""
        return np.diag(np.asarray(a, dtype=float))

    @staticmethod
    def effective_state(a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """x~ = A * x = a (elementwise) * x."""
        return np.asarray(a, dtype=float) * np.asarray(x, dtype=float)

    def metric(self, a: np.ndarray) -> np.ndarray:
        """W(a) = sum_k a_k * W^(k)."""
        a = np.asarray(a, dtype=float)
        if a.shape[0] != len(self.domain_metrics):
            raise ValueError(
                f"activation vector length {a.shape[0]} does not match "
                f"{len(self.domain_metrics)} registered domain metrics"
            )
        shape = self.domain_metrics[0].shape
        w = np.zeros(shape)
        for a_k, w_k in zip(a, self.domain_metrics, strict=True):
            w += a_k * w_k
        return w

    def compose(self, other: DomainNullification) -> DomainNullification:
        """Combine this domain set with `other`'s into one wider domain set.

        Concatenates both domain-metric registries (this domain's metrics
        first, then `other`'s), letting a room activate across domains
        registered in either source -- e.g. merging a "structural" domain
        catalogue with a "control" domain catalogue into one unified
        engineering-space geometry, per the doc's "Engineering Spaces"
        hierarchy (Structural, Mechanical, Energy, ...).
        """
        combined = list(self.domain_metrics) + list(other.domain_metrics)
        return DomainNullification(combined)
