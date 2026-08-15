"""Intelligent compute allocation (docs/QES-architecture.md, sections 20-21).

    Priority_i = (pi_i + eps)^alpha * (U_i + eps)^beta * (1 + Risk_i)^gamma * V_i^delta
    rho_i      = R_total * Priority_i / sum_j Priority_j

    Hardware resource constraint: sum_i c_ir <= C_r for each resource type r.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class RoomComputeProfile:
    """Inputs to the priority formula for a single room."""

    permission: float  # pi_i
    uncertainty: float  # U_i
    risk: float  # Risk_i
    value: float  # V_i


class ComputeAllocator:
    """Allocates a total compute budget across rooms according to Priority_i."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        epsilon: float = 1e-6,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def priority(self, profile: RoomComputeProfile) -> float:
        """Priority_i = (pi+eps)^alpha * (U+eps)^beta * (1+Risk)^gamma * V^delta."""
        v = max(profile.value, self.epsilon)
        return (
            (profile.permission + self.epsilon) ** self.alpha
            * (profile.uncertainty + self.epsilon) ** self.beta
            * (1.0 + profile.risk) ** self.gamma
            * v ** self.delta
        )

    def allocate(
        self, profiles: Sequence[RoomComputeProfile], total: float
    ) -> np.ndarray:
        """rho_i = total * Priority_i / sum_j Priority_j."""
        priorities = np.array([self.priority(p) for p in profiles], dtype=float)
        denom = priorities.sum()
        if denom <= 0:
            return np.zeros_like(priorities)
        return total * priorities / denom

    @staticmethod
    def check_resource_constraint(requested: np.ndarray, capacity: float) -> bool:
        """True iff sum_i c_ir <= C_r for a single resource type r."""
        return bool(np.sum(requested) <= capacity)

    @staticmethod
    def clip_to_capacity(requested: np.ndarray, capacity: float) -> np.ndarray:
        """Scale down proportionally if requested demand exceeds capacity C_r."""
        total = np.sum(requested)
        if total <= capacity or total == 0:
            return requested
        return requested * (capacity / total)

    def allocate_with_floor(
        self,
        profiles: Sequence[RoomComputeProfile],
        total: float,
        min_share: float = 0.0,
    ) -> np.ndarray:
        """Priority-weighted allocation with a guaranteed minimum fair share.

        Every room first receives `min_share * total / n` (a fairness
        floor, e.g. so low-priority/exploratory rooms are never fully
        starved), then the remaining `(1 - min_share) * total` budget is
        distributed across rooms proportional to Priority_i as in
        `allocate()`. `min_share` in [0, 1]; 0 reduces to plain
        priority-proportional `allocate()`, 1 reduces to an equal split.
        """
        n = len(profiles)
        if n == 0:
            return np.zeros(0)
        min_share = float(np.clip(min_share, 0.0, 1.0))
        floor = np.full(n, min_share * total / n)
        remaining = total * (1.0 - min_share)
        proportional = self.allocate(profiles, remaining)
        return floor + proportional

    def allocate_multi_resource(
        self,
        profiles: Sequence[RoomComputeProfile],
        capacities: dict,
    ) -> dict:
        """Allocate several independently constrained resource types at once.

        Args:
            profiles: per-room compute profiles (shared priorities across
                all resource types, i.e. rho_i is computed once and applied
                proportionally to every resource's own total capacity).
            capacities: mapping resource name -> total capacity C_r.

        Returns:
            dict mapping resource name -> np.ndarray of per-room allocations,
            each individually satisfying `sum_i c_ir <= C_r`.
        """
        return {
            resource: self.allocate(profiles, capacity)
            for resource, capacity in capacities.items()
        }
