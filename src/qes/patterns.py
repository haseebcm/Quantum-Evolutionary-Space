"""PatternS — generative memory (docs/QES-architecture.md, section 28).

    g_k = P_k(intent, context)

Selected patterns improve over time (Phi down, CCI down, M up). A successful
virtual architecture becomes Pattern_j; future rooms can start from it rather
than searching from zero:

    Simulation -> Validation -> Pattern Memory -> Future Generation
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_pattern_id_counter = itertools.count(1)


def _next_pattern_id() -> str:
    return f"P-{next(_pattern_id_counter):05d}"


def _validate_finite_metric(name: str, value: float) -> float:
    metric = float(value)
    if not np.isfinite(metric):
        raise ValueError(f"{name} must be finite")
    return metric


@dataclass
class Pattern:
    """A stored, reusable virtual architecture / configuration."""

    intent: str
    context: dict[str, Any]
    payload: Any  # e.g. equation population, activation vector, theta, etc.
    phi: float = 0.0
    cci: float = 0.0
    margin: float = 0.0
    uses: int = 0
    id: str = field(default_factory=_next_pattern_id)

    def __post_init__(self) -> None:
        if not isinstance(self.intent, str):
            raise TypeError("intent must be a string")
        if not self.intent:
            raise ValueError("intent must be non-empty")
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")
        self.context = dict(self.context)
        self.phi = _validate_finite_metric("phi", self.phi)
        self.cci = _validate_finite_metric("cci", self.cci)
        self.margin = _validate_finite_metric("margin", self.margin)
        if not isinstance(self.uses, int) or isinstance(self.uses, bool):
            raise TypeError("uses must be an integer")
        if self.uses < 0:
            raise ValueError("uses must be >= 0")
        if not isinstance(self.id, str):
            raise TypeError("id must be a string")
        if not self.id:
            raise ValueError("id must be non-empty")

    def improves_on(self, other: Pattern) -> bool:
        """True iff this pattern is no worse on Phi/CCI/margin and strictly
        better on at least one axis."""
        if not isinstance(other, Pattern):
            raise TypeError("other must be a Pattern")
        no_worse = self.phi <= other.phi and self.cci <= other.cci and self.margin >= other.margin
        strictly_better = (
            self.phi < other.phi or self.cci < other.cci or self.margin > other.margin
        )
        return no_worse and strictly_better


class PatternMemory:
    """Stores and retrieves patterns keyed by intent, ranked by quality."""

    def __init__(self) -> None:
        self._patterns: dict[str, list[Pattern]] = {}

    def store(self, pattern: Pattern) -> None:
        """Store `pattern` under its intent, replacing an existing entry with
        the same id to avoid duplicate memory rows."""
        if not isinstance(pattern, Pattern):
            raise TypeError("pattern must be a Pattern")
        patterns = self._patterns.setdefault(pattern.intent, [])
        for index, existing in enumerate(patterns):
            if existing.id == pattern.id:
                patterns[index] = pattern
                return
        patterns.append(pattern)

    def generate(self, intent: str, context: dict[str, Any]) -> Pattern | None:
        """g_k = P_k(intent, context): retrieve the best matching stored
        pattern, preferring entries whose stored context is satisfied by the
        requested `context`."""
        if not isinstance(intent, str):
            raise TypeError("intent must be a string")
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")

        candidates = self._patterns.get(intent, [])
        if not candidates:
            return None

        query_context = dict(context)
        matching = [
            pattern
            for pattern in candidates
            if all(query_context.get(key) == value for key, value in pattern.context.items())
        ]
        ranked_candidates = matching if matching else candidates
        best = min(ranked_candidates, key=lambda p: (p.phi, p.cci, -p.margin, p.uses, p.id))
        best.uses += 1
        return best

    def retire_dominated(self, intent: str) -> None:
        """Drop patterns dominated by a strictly better pattern of the same intent."""
        if not isinstance(intent, str):
            raise TypeError("intent must be a string")

        candidates = self._patterns.get(intent, [])
        if len(candidates) < 2:
            return

        phi = np.asarray([pattern.phi for pattern in candidates], dtype=float)
        cci = np.asarray([pattern.cci for pattern in candidates], dtype=float)
        margin = np.asarray([pattern.margin for pattern in candidates], dtype=float)

        phi_le = phi[:, None] <= phi[None, :]
        cci_le = cci[:, None] <= cci[None, :]
        margin_ge = margin[:, None] >= margin[None, :]
        strict = (phi[:, None] < phi[None, :]) | (cci[:, None] < cci[None, :]) | (
            margin[:, None] > margin[None, :]
        )
        dominates = phi_le & cci_le & margin_ge & strict
        dominated = np.any(dominates, axis=0)

        self._patterns[intent] = [
            pattern for pattern, is_dominated in zip(candidates, dominated, strict=True) if not is_dominated
        ]

    def all_patterns(self, intent: str | None = None) -> list[Pattern]:
        """Return all stored patterns, optionally filtered by `intent`."""
        if intent is not None:
            if not isinstance(intent, str):
                raise TypeError("intent must be a string")
            return list(self._patterns.get(intent, []))
        return [pattern for patterns in self._patterns.values() for pattern in patterns]
