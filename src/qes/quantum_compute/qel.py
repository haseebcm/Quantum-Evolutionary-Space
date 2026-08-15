"""QEL stream wrapper: manages a single evolution stream's history.

Tracks deltas, generation depth, and provides simple lineage metadata. This
maps to the QEL idea in the architecture: an isolated evolution layer with
its own state, trace, and delta history.
"""
from __future__ import annotations

import copy as pycopy
import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QELStream:
    name: str
    current: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    depth: int = 0
    deltas: list[np.ndarray] = field(default_factory=list)
    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    generation_id: int = 0
    birth_time: float = field(default_factory=time.time)
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    def apply_delta(self, delta: np.ndarray) -> None:
        """Apply a delta to the current state.

        Parameters
        ----------
        delta : np.ndarray
            The delta to apply.
        """
        self.current = np.asarray(self.current, dtype=float) + np.asarray(delta, dtype=float)
        if self.bounds is not None:
            self.current = np.clip(self.current, self.bounds[0], self.bounds[1])
        self.deltas.append(np.asarray(delta, dtype=float))
        self.depth += 1

    def apply_deltas(self, deltas: list[np.ndarray]) -> None:
        """Apply multiple deltas sequentially.

        Parameters
        ----------
        deltas : list[np.ndarray]
            A list of deltas to apply.
        """
        for delta in deltas:
            self.apply_delta(delta)

    def reset(self) -> None:
        """Reset the stream to its zero state, clearing deltas and depth."""
        self.current = np.zeros_like(self.current)
        self.depth = 0
        self.deltas.clear()

    def snapshot(self) -> dict:
        """Create a summary snapshot of the stream.

        Returns
        -------
        dict
            Dictionary containing key stream metadata and current state.
        """
        return {
            "name": self.name,
            "stream_id": self.stream_id,
            "depth": self.depth,
            "current": self.current.copy(),
            "deltas_count": len(self.deltas),
            "generation_id": self.generation_id,
        }

    def branch(self, name: str | None = None) -> QELStream:
        """Create a child stream with shared lineage and a copy of the state.

        Parameters
        ----------
        name : str | None
            Optional name for the branched stream.

        Returns
        -------
        QELStream
            A new branched child stream.
        """
        child_name = name if name else f"{self.name}_branch_{len(self.children)}"
        child = QELStream(
            name=child_name,
            current=self.current.copy(),
            depth=self.depth,
            deltas=self.deltas.copy(),
            parent_id=self.stream_id,
            generation_id=self.generation_id + 1,
            bounds=pycopy.deepcopy(self.bounds) if self.bounds else None,
        )
        self.children.append(child.stream_id)
        return child

    def merge(self, other: QELStream, strategy: str = "average") -> None:
        """Merge another stream into this one.

        Parameters
        ----------
        other : QELStream
            The other stream to merge.
        strategy : str
            Strategy to use: 'average', 'replace', or 'weighted'.
        """
        if strategy == "average":
            self.current = (self.current + other.current) / 2.0
        elif strategy == "replace":
            self.current = other.current.copy()
        elif strategy == "weighted":
            total_depth = self.depth + other.depth
            if total_depth > 0:
                w1 = self.depth / total_depth
                w2 = other.depth / total_depth
            else:
                w1, w2 = 0.5, 0.5
            self.current = self.current * w1 + other.current * w2
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        if self.bounds is not None:
            self.current = np.clip(self.current, self.bounds[0], self.bounds[1])

    def rollback(self, n: int = 1) -> None:
        """Undo the last n deltas.

        Parameters
        ----------
        n : int
            Number of deltas to undo.
        """
        n_actual = min(n, len(self.deltas))
        for _ in range(n_actual):
            delta = self.deltas.pop()
            self.current -= delta
            self.depth -= 1

        if self.bounds is not None:
            self.current = np.clip(self.current, self.bounds[0], self.bounds[1])

    def delta_norm_history(self) -> list[float]:
        """Get the L2 norm history of all deltas.

        Returns
        -------
        list[float]
            List of L2 norms of the applied deltas.
        """
        return [float(np.linalg.norm(d)) for d in self.deltas]

    def mean_delta(self) -> np.ndarray:
        """Calculate the average delta.

        Returns
        -------
        np.ndarray
            Mean of all applied deltas.
        """
        if not self.deltas:
            return np.zeros_like(self.current)
        return np.mean(self.deltas, axis=0)

    def delta_variance(self) -> float:
        """Calculate the variance of delta norms.

        Returns
        -------
        float
            Variance of the delta norms.
        """
        if not self.deltas:
            return 0.0
        norms = self.delta_norm_history()
        return float(np.var(norms))

    def is_converging(self, window: int = 5, atol: float = 1e-6) -> bool:
        """Check if the stream is converging based on recent delta norms.

        Parameters
        ----------
        window : int
            Number of recent deltas to consider.
        atol : float
            Absolute tolerance to consider converged.

        Returns
        -------
        bool
            True if converging.
        """
        if len(self.deltas) < window:
            return False
        recent_norms = self.delta_norm_history()[-window:]
        monotonic_decrease = all(
            recent_norms[i] >= recent_norms[i + 1] for i in range(len(recent_norms) - 1)
        )
        return monotonic_decrease or recent_norms[-1] <= atol

    def distance(self, other: QELStream) -> float:
        """Calculate the L2 distance to another stream's state.

        Parameters
        ----------
        other : QELStream
            The other stream to compare against.

        Returns
        -------
        float
            L2 distance.
        """
        return float(np.linalg.norm(self.current - other.current))

    def cumulative_delta(self) -> np.ndarray:
        """Get the sum of all deltas.

        Returns
        -------
        np.ndarray
            Cumulative delta applied to the stream.
        """
        if not self.deltas:
            return np.zeros_like(self.current)
        return np.sum(self.deltas, axis=0)

    def copy(self) -> QELStream:
        """Return a deep copy of the stream.

        Returns
        -------
        QELStream
            Deep copy.
        """
        return pycopy.deepcopy(self)

    def __eq__(self, other: object) -> bool:
        """Compare streams for equality by depth."""
        if not isinstance(other, QELStream):
            return NotImplemented
        return self.depth == other.depth

    def __lt__(self, other: object) -> bool:
        """Compare streams by depth."""
        if not isinstance(other, QELStream):
            return NotImplemented
        return self.depth < other.depth
