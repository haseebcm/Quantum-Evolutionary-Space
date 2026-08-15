"""DSA, DR, and HSA (docs/QES-architecture.md, sections 6-8).

DSA — Divergence-State Analysis:
    d_i(t)  = x_i(t) - x_i*
    ddot_i(t) = d/dt [x_i(t) - x_i*]
    dddot_i(t) = d^2/dt^2 d_i(t)
    D_i = (d_i, ddot_i, dddot_i)   -- position, velocity, acceleration of divergence

DR — the room-distance metric:
    DR_w = sum_j w_j * d_j^2
    DR_i = d_i^T * W(a_i) * d_i                    (domain-nullified form)
    z_ij = (x_ij - x_ij*) / sigma_ij               (standardized)
    DR_{z,i} = z_i^T * W(a_i) * z_i

HSA — Health-State Analysis / criticality geometry:
    D(t) = k * DR(t)
    S(t) = D(t) - B(t)
    S < 0   stable
    S ~= 0  critical boundary
    S > 0   singularity / limit region
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HSA_STABLE = "stable"
HSA_CRITICAL = "critical"
HSA_SINGULAR = "singular"


def hsa_state(health: float, critical_band: float = 1e-6) -> str:
    """Classify S_i(t) into {stable, critical, singular} per section 8.

    `critical_band` is the tolerance around S = 0 treated as the critical
    boundary rather than requiring an exact zero crossing (which is
    numerically fragile for continuous-valued health scores).
    """
    if health < -critical_band:
        return HSA_STABLE
    if health > critical_band:
        return HSA_SINGULAR
    return HSA_CRITICAL


@dataclass
class DivergenceResult:
    """D_i = (d_i, ddot_i, dddot_i) plus derived scalar metrics."""

    d: np.ndarray
    d_dot: np.ndarray
    d_ddot: np.ndarray
    dr: float
    health: float  # S_i(t)

    def state(self, critical_band: float = 1e-6) -> str:
        """Classify this result's health S_i(t) via `hsa_state()`."""
        return hsa_state(self.health, critical_band)


class DSA:
    """Tracks divergence position/velocity/acceleration and derived DR/HSA for a room."""

    def __init__(self):
        self._prev_d: np.ndarray | None = None
        self._prev_d_dot: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_d = None
        self._prev_d_dot = None

    def divergence(self, x: np.ndarray, x_star: np.ndarray) -> np.ndarray:
        """d_i(t) = x_i(t) - x_i*."""
        return np.asarray(x, dtype=float) - np.asarray(x_star, dtype=float)

    def update(
        self,
        x: np.ndarray,
        x_star: np.ndarray,
        dt: float,
        w: np.ndarray,
        k: float = 1.0,
        baseline: float = 0.0,
        sigma: np.ndarray | None = None,
    ) -> DivergenceResult:
        """Advance one time step and compute D_i, DR_i, and S_i(t).

        Args:
            x: current state x_i(t).
            x_star: reference state x_i*.
            dt: time step used to finite-difference velocity/acceleration.
            w: domain-nullified metric W(a_i) (n x n), or a 1-D weight vector
               (in which case DR_w = sum_j w_j * d_j^2 is used).
            k: HSA scale factor k_i.
            baseline: baseline term B_i(t) subtracted in HSA.
            sigma: optional per-component standard deviation for standardized DR.
        """
        d = self.divergence(x, x_star)

        if sigma is not None:
            z = d / np.asarray(sigma, dtype=float)
            dr_vec = z
        else:
            dr_vec = d

        if w is None:
            dr = float(np.dot(dr_vec, dr_vec))
        else:
            w_arr = np.asarray(w, dtype=float)
            if w_arr.ndim == 1:
                dr = float(np.sum(w_arr * dr_vec ** 2))
            elif w_arr.ndim == 2:
                # Check if 2D matrix is diagonal or identity
                if w_arr.shape[0] == w_arr.shape[1] and np.array_equal(w_arr, np.eye(w_arr.shape[0])):
                    dr = float(np.dot(dr_vec, dr_vec))
                else:
                    dr = float(dr_vec @ w_arr @ dr_vec)
            else:
                dr = float(dr_vec @ w_arr @ dr_vec)

        if self._prev_d is None:
            d_dot = np.zeros_like(d)
        else:
            d_dot = (d - self._prev_d) / dt

        if self._prev_d_dot is None:
            d_ddot = np.zeros_like(d)
        else:
            d_ddot = (d_dot - self._prev_d_dot) / dt

        self._prev_d = d
        self._prev_d_dot = d_dot

        health = k * dr - baseline

        return DivergenceResult(d=d, d_dot=d_dot, d_ddot=d_ddot, dr=dr, health=health)
