"""Project metrics utilities for QES and quantum_compute.

This module provides simple, well-tested numeric metrics used by the
architecture: state fidelity, purity, overlaps, gate fidelity and simple
aggregation helpers for layer vectors and weight arrays.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Return the fidelity |<a|b>|^2 between two state vectors (pure states)."""
    a = np.asarray(state_a, dtype=complex).ravel()
    b = np.asarray(state_b, dtype=complex).ravel()
    if a.size != b.size:
        raise ValueError("states must have the same dimension")
    ov = np.vdot(a, b)
    return float(np.abs(ov) ** 2)


def state_overlap(state_a: np.ndarray, state_b: np.ndarray) -> complex:
    """Return the complex inner product <a|b> (not squared).

    Useful when phase information matters.
    """
    a = np.asarray(state_a, dtype=complex).ravel()
    b = np.asarray(state_b, dtype=complex).ravel()
    if a.size != b.size:
        raise ValueError("states must have the same dimension")
    return np.vdot(a, b)


def purity(rho: np.ndarray) -> float:
    """Return purity Tr(rho^2) for a density matrix rho."""
    r = np.asarray(rho, dtype=complex)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("rho must be a square matrix")
    tr = np.trace(r @ r)
    return float(np.real_if_close(tr))


def pure_state_distance(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Return the pure-state distance sqrt(1 - |<a|b>|^2)."""
    f = fidelity(state_a, state_b)
    return float(np.sqrt(max(0.0, 1.0 - f)))


def average_gate_fidelity(U: np.ndarray, V: np.ndarray) -> float:
    """Return the average gate fidelity between unitaries U and V.

    Uses the formula: (|Tr(U^\dagger V)|^2 + d) / (d*(d+1)).
    """
    U = np.asarray(U, dtype=complex)
    V = np.asarray(V, dtype=complex)
    if U.shape != V.shape or U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U and V must be square matrices of the same shape")
    d = U.shape[0]
    t = np.trace(U.conj().T @ V)
    return float((np.abs(t) ** 2 + d) / (d * (d + 1)))


def layer_norms(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Return L2 norms for an iterable of numeric vectors."""
    arrs = [np.linalg.norm(np.asarray(v, dtype=float).ravel()) for v in vectors]
    return np.asarray(arrs, dtype=float)


def aggregate_weights_stats(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given weights shaped (n_layers, k), return (mean,k),(std,k).

    Useful for QSEE-11L per-layer weight summaries.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError("weights must be a 2D array (n_layers, k)")
    return np.mean(w, axis=0), np.std(w, axis=0)
