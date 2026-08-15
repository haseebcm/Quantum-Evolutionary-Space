"""Synchronization primitives for QSEE-11L layers.

Provide deterministic, testable synchronization helpers and a SyncManager
to coordinate merge policies and barriers across layers. These are
software-only coordination utilities and do not imply concurrent safety for
highly-parallel real-world deployments — use process/thread primitives when
needed.
"""
from __future__ import annotations

import cmath
import time
from collections.abc import Iterable
from threading import Lock

import numpy as np


class SyncManager:
    """Simple manager to coordinate deterministic merges and barriers.

    Usage: create one SyncManager and call barrier()/merge() from a single
    coordinating thread. For concurrent access, wrap calls with application
    locking as appropriate.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.history: list[dict] = []

    def barrier(self) -> None:
        """A no-op barrier placeholder — kept for API compatibility."""
        # In a distributed setting this would coordinate with remote peers.
        self.history.append({"event": "barrier", "timestamp": time.time()})
        return

    def async_barrier(self, timeout: float = 5.0) -> bool:
        """Wait at a barrier for up to timeout seconds."""
        self.history.append({"event": "async_barrier", "timeout": timeout, "timestamp": time.time()})
        # Placeholder implementation
        return True

    def merge_average(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        """Deterministically average a sequence of vectors into one vector."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        self.history.append({"event": "merge_average", "count": len(arrs), "timestamp": time.time()})
        if not arrs:
            return np.array([], dtype=float)
        return np.mean(np.vstack(arrs), axis=0)

    def merge_sync(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        """Thread-safe merge wrapper using the manager lock."""
        with self._lock:
            return self.merge_average(vectors)

    def merge_weighted(self, vectors: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
        """Merge a sequence of vectors with weights."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        w = list(weights)
        self.history.append({"event": "merge_weighted", "count": len(arrs), "timestamp": time.time()})
        if not arrs:
            return np.array([], dtype=float)
        return np.average(np.vstack(arrs), axis=0, weights=w)

    def merge_median(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        """Merge vectors by taking the component-wise median."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        self.history.append({"event": "merge_median", "count": len(arrs), "timestamp": time.time()})
        if not arrs:
            return np.array([], dtype=float)
        return np.median(np.vstack(arrs), axis=0)

    def merge_best(self, vectors: Iterable[np.ndarray], fitnesses: Iterable[float]) -> np.ndarray:
        """Return the vector with the highest fitness score."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        fits = list(fitnesses)
        self.history.append({"event": "merge_best", "count": len(arrs), "timestamp": time.time()})
        if not arrs:
            return np.array([], dtype=float)
        best_idx = int(np.argmax(fits))
        return arrs[best_idx]
        
    def consensus(self, vectors: Iterable[np.ndarray], rounds: int = 3, damping: float = 0.5) -> np.ndarray:
        """Iteratively converge towards consensus."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        self.history.append({"event": "consensus", "count": len(arrs), "rounds": rounds, "timestamp": time.time()})
        if not arrs:
            return np.array([], dtype=float)
        mat = np.vstack(arrs)
        for _ in range(rounds):
            mean_val = np.mean(mat, axis=0)
            mat = mat * (1 - damping) + mean_val * damping
        return np.mean(mat, axis=0)

    def phase_sync(self, phases: Iterable[float]) -> float:
        """Average circular phases."""
        self.history.append({"event": "phase_sync", "timestamp": time.time()})
        phases_list = list(phases)
        if not phases_list:
            return 0.0
        z = sum(cmath.rect(1.0, p) for p in phases_list)
        return cmath.phase(z)

    def detect_conflicts(self, vectors: Iterable[np.ndarray], threshold: float = 1.0) -> list[tuple[int, int, float]]:
        """Detect pairs of vectors whose distance exceeds the threshold."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        self.history.append({"event": "detect_conflicts", "count": len(arrs), "timestamp": time.time()})
        conflicts = []
        n = len(arrs)
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(arrs[i] - arrs[j]))
                if dist > threshold:
                    conflicts.append((i, j, dist))
        return conflicts

    def gossip_merge(self, local: np.ndarray, peers: Iterable[np.ndarray], alpha: float = 0.5) -> np.ndarray:
        """Merge a local vector with peer vectors using gossip protocol logic."""
        self.history.append({"event": "gossip_merge", "timestamp": time.time()})
        peer_arrs = [np.asarray(v, dtype=float).ravel() for v in peers]
        if not peer_arrs:
            return np.asarray(local, dtype=float).ravel()
        peer_mean = np.mean(np.vstack(peer_arrs), axis=0)
        return (1 - alpha) * np.asarray(local, dtype=float).ravel() + alpha * peer_mean

    def sync_quality(self, vectors: Iterable[np.ndarray]) -> float:
        """Measure the quality of synchronization (1.0 = perfect alignment)."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        self.history.append({"event": "sync_quality", "count": len(arrs), "timestamp": time.time()})
        if len(arrs) < 2:
            return 1.0
        n = len(arrs)
        total_dist = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += float(np.linalg.norm(arrs[i] - arrs[j]))
                pairs += 1
        mean_dist = total_dist / pairs
        return float(np.exp(-mean_dist))

    def merge_top_k(self, vectors: Iterable[np.ndarray], fitnesses: Iterable[float], k: int) -> np.ndarray:
        """Merge the best k vectors according to fitness scores."""
        arrs = [np.asarray(v, dtype=float).ravel() for v in vectors]
        fits = list(fitnesses)
        self.history.append({"event": "merge_top_k", "count": len(arrs), "k": k, "timestamp": time.time()})
        if not arrs:
            return np.array([], dtype=float)
        k = min(k, len(arrs))
        if k == 0:
            return np.array([], dtype=float)
        top_k_indices = np.argsort(fits)[-k:]
        top_k_vectors = [arrs[i] for i in top_k_indices]
        return np.mean(np.vstack(top_k_vectors), axis=0)
