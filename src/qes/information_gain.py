"""Phase 8 -- Information Gain Engine.

This module asks a classical active-search question inside QES: if several
candidate experiment branches are available, which one is expected to reduce
the population's uncertainty the most?

    IG(a) = H(before) - H(after | a)

Entropy is not redefined here: the implementation deliberately reuses
`qes.convergence.qes_entropy` so QES keeps one Shannon-entropy convention
across convergence tracking and information-gain scoring. The resulting
scores can feed room/branch prioritization without duplicating the existing
room-level compute allocator's role.

Everything here is ordinary CPU/RAM classical computation over room weights.
"Information gain" means reduced uncertainty in a Python/Numpy search
population, not literal quantum measurement and not physical universe
creation.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import isclose, isfinite

from qes.compute_allocator import ComputeAllocator, RoomComputeProfile
from qes.convergence import qes_entropy
from qes.room import Room


def _validate_real_number(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_entropy(name: str, value: float) -> float:
    entropy = _validate_real_number(name, value)
    if entropy < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return entropy


def _validate_candidate_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("candidate names must be strings")
    if not name:
        raise ValueError("candidate names must be non-empty")
    return name


def _validate_rooms(name: str, rooms: Sequence[Room]) -> list[Room]:
    if isinstance(rooms, (str, bytes)) or not isinstance(rooms, Sequence):
        raise TypeError(f"{name} must be a sequence of Room instances")
    validated = list(rooms)
    if not validated:
        raise ValueError(f"{name} must contain at least one Room")
    for room in validated:
        if not isinstance(room, Room):
            raise TypeError(f"{name} must contain only Room instances")
        weight = _validate_real_number(f"{name} room weight", room.weight)
        if weight < 0.0:
            raise ValueError(f"{name} room weights must be >= 0")
    return validated


def information_gain(uncertainty_before: float, uncertainty_after: float) -> float:
    """Return ``H(before) - H(after)``.

    A negative result is valid: it means the action increased uncertainty
    instead of reducing it, so the branch taught QES less than expected or
    actively made the belief state more diffuse.
    """
    before = _validate_entropy("uncertainty_before", uncertainty_before)
    after = _validate_entropy("uncertainty_after", uncertainty_after)
    return before - after


def population_entropy(rooms: list[Room]) -> float:
    """Compute population entropy from room weights via `qes_entropy`."""
    validated = _validate_rooms("rooms", rooms)
    weights = [room.weight for room in validated]
    return qes_entropy(weights)


class InformationGainEstimator:
    """Estimates realized and expected information gain across branches."""

    def estimate_realized(self, rooms_before: list[Room], rooms_after: list[Room]) -> float:
        """Measure actual information gain between two room populations."""
        before_entropy = population_entropy(rooms_before)
        after_entropy = population_entropy(rooms_after)
        return information_gain(before_entropy, after_entropy)

    def estimate_expected(
        self,
        candidates: dict[str, tuple[list[Room], list[Room]]],
    ) -> dict[str, float]:
        """Estimate information gain for each named candidate branch."""
        if not isinstance(candidates, dict):
            raise TypeError("candidates must be a dict[str, tuple[list[Room], list[Room]]]")
        estimates: dict[str, float] = {}
        for candidate_name, populations in candidates.items():
            _validate_candidate_name(candidate_name)
            if not isinstance(populations, tuple) or len(populations) != 2:
                raise TypeError(
                    "each candidate value must be a tuple of (rooms_before, rooms_after)"
                )
            rooms_before, rooms_after = populations
            estimates[candidate_name] = self.estimate_realized(rooms_before, rooms_after)
        return estimates


def rank_by_information_gain(candidates: dict[str, float]) -> list[tuple[str, float]]:
    """Sort candidate scores by descending information gain, then by name."""
    if not isinstance(candidates, dict):
        raise TypeError("candidates must be a dict[str, float]")
    ranked: list[tuple[str, float]] = []
    for candidate_name, score in candidates.items():
        ranked.append(
            (
                _validate_candidate_name(candidate_name),
                _validate_real_number(f"information gain for {candidate_name!r}", score),
            )
        )
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def allocate_by_information_gain(
    ig_scores: dict[str, float],
    total_budget: float,
    min_floor: float = 0.0,
) -> dict[str, float]:
    """Allocate compute from candidate-level information-gain scores.

    Each candidate first receives an absolute floor of `min_floor`. The
    remaining budget is then distributed proportional to the positive part of
    each candidate's information-gain score. If every score is non-positive,
    the remaining budget is split equally because there is no positive signal
    to prefer one branch over another.

    The proportional part reuses `ComputeAllocator`'s existing
    priority-normalization pattern, but this helper stays branch-oriented
    because it works on named candidate experiments rather than on rooms and
    because it needs an absolute per-candidate floor plus an equal-split
    fallback when all scores are non-positive.
    """
    if not isinstance(ig_scores, dict):
        raise TypeError("ig_scores must be a dict[str, float]")

    budget = _validate_real_number("total_budget", total_budget)
    if budget < 0.0:
        raise ValueError("total_budget must be >= 0")

    floor = _validate_real_number("min_floor", min_floor)
    if floor < 0.0:
        raise ValueError("min_floor must be >= 0")

    candidate_names: list[str] = []
    sanitized_scores: dict[str, float] = {}
    for candidate_name, score in ig_scores.items():
        name = _validate_candidate_name(candidate_name)
        candidate_names.append(name)
        sanitized_scores[name] = _validate_real_number(
            f"information gain for {candidate_name!r}",
            score,
        )

    count = len(candidate_names)
    if count == 0:
        if budget == 0.0:
            return {}
        raise ValueError("cannot allocate a positive budget with no candidates")

    required_floor = floor * count
    if required_floor > budget and not isclose(required_floor, budget, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("min_floor is infeasible for the given total_budget and candidate count")

    remaining = budget - required_floor
    allocations = {name: floor for name in candidate_names}
    if isclose(remaining, 0.0, rel_tol=0.0, abs_tol=1e-12):
        total_allocated = sum(allocations.values())
        if not isclose(total_allocated, budget, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError("allocation does not sum to total_budget")
        return allocations

    positive_scores = {name: max(score, 0.0) for name, score in sanitized_scores.items()}
    positive_total = sum(positive_scores.values())

    if positive_total > 0.0:
        allocator = ComputeAllocator(alpha=0.0, beta=0.0, gamma=0.0, delta=1.0, epsilon=0.0)
        profiles = [
            RoomComputeProfile(permission=1.0, uncertainty=1.0, risk=0.0, value=positive_scores[name])
            for name in candidate_names
        ]
        proportional = allocator.allocate(profiles, remaining)
        for name, amount in zip(candidate_names, proportional, strict=True):
            allocations[name] += float(amount)
    else:
        equal_share = remaining / count
        for name in candidate_names:
            allocations[name] += equal_share

    total_allocated = sum(allocations.values())
    if not isclose(total_allocated, budget, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError("allocation does not sum to total_budget")
    return allocations


__all__ = [
    "InformationGainEstimator",
    "allocate_by_information_gain",
    "information_gain",
    "population_entropy",
    "rank_by_information_gain",
]
