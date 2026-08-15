"""Digital Twin anchor (docs/QES-architecture.md, sections 22-23).

    e_i^Twin(t) = y(t) - yhat_i(t)
    delta_i     = (e_i^Twin)^T * R^-1 * e_i^Twin

Sequential evidence update (Bayesian-style room-weight update):
    p_i^(t+1) = p_i^t * L(y_{t+1} | R_i) / sum_j [ p_j^t * L(y_{t+1} | R_j) ]
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class DigitalTwin:
    """Anchors virtual rooms to an observed real system and updates their belief weights."""

    def __init__(self, noise_covariance: np.ndarray | None = None, window: int = 20):
        """
        Args:
            noise_covariance: R, the observation noise covariance matrix
                used in delta_i.
            window: size of the rolling residual-history window used by
                `record_residual()`/`detect_drift()`.
        """
        self.noise_covariance = noise_covariance
        self.window = window
        self._residual_history: list = []

    def twin_residual(self, observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """e_i^Twin(t) = y(t) - yhat_i(t)."""
        return np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)

    def weighted_twin_error(self, residual: np.ndarray) -> float:
        """delta_i = e^T * R^-1 * e."""
        residual = np.asarray(residual, dtype=float)
        if self.noise_covariance is None:
            r_inv = np.eye(residual.shape[0])
        else:
            r_inv = np.linalg.inv(self.noise_covariance)
        return float(residual @ r_inv @ residual)

    @staticmethod
    def likelihood_gaussian(residual: np.ndarray, covariance: np.ndarray) -> float:
        """Gaussian likelihood L(y | R_i) given residual e and covariance Sigma."""
        residual = np.asarray(residual, dtype=float)
        n = residual.shape[0]
        cov = np.asarray(covariance, dtype=float)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        norm = 1.0 / np.sqrt((2 * np.pi) ** n * max(det, 1e-300))
        exponent = -0.5 * float(residual @ inv @ residual)
        return norm * np.exp(exponent)

    @staticmethod
    def evidence_update(weights: Sequence[float], likelihoods: Sequence[float]) -> np.ndarray:
        """p_i^(t+1) = p_i^t * L_i / sum_j (p_j^t * L_j)."""
        w = np.asarray(weights, dtype=float)
        lik = np.asarray(likelihoods, dtype=float)
        numerators = w * lik
        denom = numerators.sum()
        if denom <= 0:
            # No evidence discriminates between rooms; keep weights unchanged (renormalized).
            total = w.sum()
            return w / total if total > 0 else w
        return numerators / denom

    def record_residual(self, delta: float) -> None:
        """Append a weighted-error observation `delta_i` to the rolling window."""
        self._residual_history.append(delta)
        if len(self._residual_history) > self.window:
            del self._residual_history[0]

    def rolling_mean_error(self) -> float:
        """Mean of `delta_i` over the trailing `window` observations."""
        if not self._residual_history:
            return 0.0
        return float(np.mean(self._residual_history))

    def detect_drift(self, threshold: float) -> bool:
        """True iff the rolling mean weighted twin error exceeds `threshold`.

        A sustained rise in `delta_i` (the twin's weighted residual, section
        22-23) signals that the virtual room has drifted away from the
        observed real system -- this flags when a room should be
        re-anchored, re-mutated, or collapsed rather than trusted further.
        """
        return self.rolling_mean_error() > threshold
