"""Synchronization and gate-like utilities for simulated layers.

These utilities model coordination primitives (e.g., no-cross-layer writes,
controlled synchronization points) as simple helper functions. They are
software-only and do not imply any physical gate semantics.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def enforce_isolation(layers_vectors: Iterable[np.ndarray]) -> bool:
    """Validate that layers' vectors do not share mutable references.

    This function is a lightweight sanity-check helper; it does not prevent
    concurrent mutation in multi-threaded contexts by itself.
    """
    # Quick heuristic: check that base object ids differ
    ids = [id(v) for v in layers_vectors]
    return len(ids) == len(set(ids))


def sync_average(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Return the per-dimension average across provided vectors.

    Useful as a conservative synchronization/merge policy when a higher-level
    coordinator needs a deterministic combined signal.
    """
    arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    return np.mean(np.vstack(arrs), axis=0)


def sync_weighted(vectors: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    """Return the per-dimension weighted average across provided vectors.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The vectors to combine.
    weights : Iterable[float]
        The weights corresponding to each vector.

    Returns
    -------
    np.ndarray
        The weighted average of the vectors.
    """
    arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    w = list(weights)
    return np.average(np.vstack(arrs), axis=0, weights=w)


def sync_median(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Return the per-dimension median across provided vectors.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The vectors to combine.

    Returns
    -------
    np.ndarray
        The median of the vectors.
    """
    arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    return np.median(np.vstack(arrs), axis=0)


def sync_max_norm(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Select the vector with the largest L2 norm.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The vectors to evaluate.

    Returns
    -------
    np.ndarray
        The vector with the largest L2 norm.
    """
    arrs = [np.asarray(v, dtype=float) for v in vectors]
    if not arrs:
        raise ValueError("vectors must not be empty.")
    norms = [np.linalg.norm(a) for a in arrs]
    return arrs[np.argmax(norms)]


def gated_sync(vectors: Iterable[np.ndarray], permissions: Iterable[bool]) -> np.ndarray:
    """Average only admitted vectors based on permissions.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The candidate vectors to combine.
    permissions : Iterable[bool]
        Boolean flags indicating whether each vector is admitted.

    Returns
    -------
    np.ndarray
        The average of the admitted vectors.
    """
    arrs = [
        np.asarray(v, dtype=float).ravel()
        for v, p in zip(vectors, permissions, strict=False)
        if p
    ]
    if not arrs:
        raise ValueError("No vectors admitted.")
    return np.mean(np.vstack(arrs), axis=0)


def isolation_report(layers_vectors: Iterable[np.ndarray]) -> dict[str, Any]:
    """Generate a detailed report with isolation violation details.

    Parameters
    ----------
    layers_vectors : Iterable[np.ndarray]
        The layers' vectors to check for shared references.

    Returns
    -------
    dict
        A dictionary containing the total count, unique count, and whether isolation is maintained.
    """
    ids = [id(v) for v in layers_vectors]
    unique_ids = set(ids)
    is_isolated = len(ids) == len(unique_ids)
    return {
        "is_isolated": is_isolated,
        "total_vectors": len(ids),
        "unique_vectors": len(unique_ids),
        "violation_count": len(ids) - len(unique_ids)
    }


def cross_correlation(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Compute the NxN cross-correlation matrix of the provided vectors.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The vectors to correlate.

    Returns
    -------
    np.ndarray
        An NxN correlation matrix.
    """
    arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    if not arrs:
        return np.array([[]])
    return np.corrcoef(np.vstack(arrs))


def sync_bounded(vectors: Iterable[np.ndarray], low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clamp the average of the vectors within provided bounds.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The vectors to average.
    low : np.ndarray
        The lower bounds for clamping.
    high : np.ndarray
        The upper bounds for clamping.

    Returns
    -------
    np.ndarray
        The clamped average of the vectors.
    """
    avg = sync_average(vectors)
    return np.clip(avg, low, high)


def sync_momentum(current: np.ndarray, target: np.ndarray, momentum: float = 0.9) -> np.ndarray:
    """Apply an exponential moving average update towards a target vector.

    Parameters
    ----------
    current : np.ndarray
        The current vector state.
    target : np.ndarray
        The target vector state.
    momentum : float, optional
        The momentum factor (default is 0.9).

    Returns
    -------
    np.ndarray
        The updated vector.
    """
    curr = np.asarray(current, dtype=float)
    tgt = np.asarray(target, dtype=float)
    return momentum * curr + (1 - momentum) * tgt


def sync_divergence(vectors: Iterable[np.ndarray]) -> float:
    """Compute the mean pairwise L2 distance among the vectors.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The vectors to measure.

    Returns
    -------
    float
        The mean pairwise L2 distance.
    """
    arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    n = len(arrs)
    if n < 2:
        return 0.0
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += float(np.linalg.norm(arrs[i] - arrs[j]))
            count += 1
    return total_dist / count


def sync_top_k(vectors: Iterable[np.ndarray], fitnesses: Iterable[float], k: int) -> np.ndarray:
    """Average the top-k vectors sorted by their fitness scores.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        The candidate vectors.
    fitnesses : Iterable[float]
        The fitness score for each vector.
    k : int
        The number of top vectors to average.

    Returns
    -------
    np.ndarray
        The average of the top-k vectors.
    """
    arrs = list(vectors)
    fits = list(fitnesses)
    if k > len(arrs):
        k = len(arrs)
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    # Sort descending by fitness
    sorted_pairs = sorted(zip(fits, arrs, strict=False), key=lambda x: x[0], reverse=True)
    top_k_arrs = [pair[1] for pair in sorted_pairs[:k]]
    return sync_average(top_k_arrs)
