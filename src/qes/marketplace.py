"""Phase 7 -- reality marketplace / resource economy.

This module builds a broader market layer on top of QES's earlier allocation
helpers. Phase 6's `ResourceNegotiation` handled one resource type at a time
between rooms. Phase 7 generalizes that idea into a multi-resource marketplace
where each reality can bid for compute, memory, and energy simultaneously while
justifying the request with expected utility, risk, novelty, uncertainty, and
information gain.

The implementation is deliberately classical. These "realities" are ordinary
Python dataclasses competing for finite CPU/RAM-style budgets inside one
process, not literal universes and not quantum hardware.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from qes.compute_allocator import ComputeAllocator, RoomComputeProfile
from qes.information_gain import information_gain

RESOURCE_NAMES = ("compute", "memory", "energy")
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "utility": 0.35,
    "risk": 0.20,
    "novelty": 0.10,
    "uncertainty": 0.15,
    "information_gain": 0.20,
}
_RESOURCE_BUDGET_FIELDS = {
    "compute": "compute_budget",
    "memory": "memory_budget",
    "energy": "energy_budget",
}
_EPSILON = 1e-12


def _validate_identifier(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _validate_real_number(name: str, value: float, *, allow_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if not allow_negative and numeric < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return numeric


def _validate_resource_name(resource: str) -> str:
    if not isinstance(resource, str):
        raise TypeError("resource names must be strings")
    if resource not in RESOURCE_NAMES:
        raise ValueError(f"resource must be one of {list(RESOURCE_NAMES)}, got {resource!r}")
    return resource


def _validate_resource_amounts(
    name: str,
    resource_amounts: Mapping[str, float],
    *,
    allow_empty: bool = False,
) -> dict[str, float]:
    if not isinstance(resource_amounts, Mapping):
        raise TypeError(f"{name} must be a mapping of resource name -> amount")
    validated: dict[str, float] = {}
    for resource, amount in resource_amounts.items():
        validated[_validate_resource_name(resource)] = _validate_real_number(
            f"{name}[{resource!r}]",
            amount,
        )
    if not allow_empty and not validated:
        raise ValueError(f"{name} must contain at least one resource amount")
    return validated


def _resolve_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_SCORE_WEIGHTS)
    if weights is None:
        return resolved
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping of score component -> weight")
    for name, value in weights.items():
        if name not in DEFAULT_SCORE_WEIGHTS:
            raise KeyError(f"unsupported weight name: {name!r}")
        resolved[name] = _validate_real_number(f"weights[{name!r}]", value)
    if sum(resolved.values()) <= 0.0:
        raise ValueError("at least one score weight must be > 0")
    return resolved


@dataclass
class RealityAccount:
    """Per-reality market state and hard per-round resource budgets."""

    reality_id: str
    compute_budget: float
    memory_budget: float
    energy_budget: float
    priority: float = 1.0
    risk: float = 0.0
    expected_utility: float = 0.0
    novelty: float = 0.0
    historical_performance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reality_id = _validate_identifier("reality_id", self.reality_id)
        self.compute_budget = _validate_real_number("compute_budget", self.compute_budget)
        self.memory_budget = _validate_real_number("memory_budget", self.memory_budget)
        self.energy_budget = _validate_real_number("energy_budget", self.energy_budget)
        self.priority = _validate_real_number("priority", self.priority)
        self.risk = _validate_real_number("risk", self.risk)
        self.expected_utility = _validate_real_number("expected_utility", self.expected_utility)
        self.novelty = _validate_real_number("novelty", self.novelty)
        self.historical_performance = _validate_real_number(
            "historical_performance",
            self.historical_performance,
        )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    def resource_budget(self, resource: str) -> float:
        """Return this account's hard cap for one market resource."""
        resource_name = _validate_resource_name(resource)
        return float(getattr(self, _RESOURCE_BUDGET_FIELDS[resource_name]))


@dataclass
class MarketBid:
    """A multi-resource request backed by market-facing quality signals."""

    reality_id: str
    requested_resources: dict[str, float]
    uncertainty: float
    information_gain: float = 0.0
    expected_utility: float | None = None
    risk: float | None = None
    novelty: float | None = None
    max_allocation: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reality_id = _validate_identifier("reality_id", self.reality_id)
        self.requested_resources = _validate_resource_amounts("requested_resources", self.requested_resources)
        if all(amount <= 0.0 for amount in self.requested_resources.values()):
            raise ValueError("requested_resources must contain at least one positive request")
        self.uncertainty = _validate_real_number("uncertainty", self.uncertainty)
        self.information_gain = _validate_real_number(
            "information_gain",
            self.information_gain,
            allow_negative=True,
        )
        if self.expected_utility is not None:
            self.expected_utility = _validate_real_number(
                "expected_utility",
                self.expected_utility,
            )
        if self.risk is not None:
            self.risk = _validate_real_number("risk", self.risk)
        if self.novelty is not None:
            self.novelty = _validate_real_number("novelty", self.novelty)
        if self.max_allocation is not None:
            validated_caps = _validate_resource_amounts("max_allocation", self.max_allocation)
            extra_keys = set(validated_caps) - set(self.requested_resources)
            if extra_keys:
                raise ValueError("max_allocation keys must be a subset of requested_resources")
            self.max_allocation = validated_caps
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    @classmethod
    def from_entropy_change(
        cls,
        *,
        reality_id: str,
        requested_resources: dict[str, float],
        uncertainty_before: float,
        uncertainty_after: float,
        uncertainty: float,
        expected_utility: float | None = None,
        risk: float | None = None,
        novelty: float | None = None,
        max_allocation: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MarketBid:
        """Build a bid whose information-gain term comes from Phase 8's helper."""
        return cls(
            reality_id=reality_id,
            requested_resources=requested_resources,
            uncertainty=uncertainty,
            information_gain=information_gain(uncertainty_before, uncertainty_after),
            expected_utility=expected_utility,
            risk=risk,
            novelty=novelty,
            max_allocation=max_allocation,
            metadata=dict(metadata or {}),
        )


def compute_allocation_score(
    expected_utility: float,
    risk: float,
    novelty: float,
    uncertainty: float,
    information_gain_value: float,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return the core Phase 7 compute-allocation score.

    The roadmap asks for::

        ComputeAllocation = f(Utility, Risk, Novelty, Uncertainty, InformationGain)

    This implementation makes that explicit as a weighted linear score with
    risk acting as a penalty term::

        raw_score = (
            w_utility * expected_utility
            + w_novelty * novelty
            + w_uncertainty * uncertainty
            + w_information_gain * information_gain
            - w_risk * risk
        )
        score = max(0.0, raw_score)

    Higher utility, novelty, uncertainty, or information gain therefore
    increase a bid's competitiveness, while higher risk reduces it. The final
    `max(0.0, ...)` floor keeps the score non-negative so the marketplace can
    use it safely as a proportional allocation signal.
    """
    resolved = _resolve_weights(weights)
    utility = _validate_real_number("expected_utility", expected_utility)
    risk_value = _validate_real_number("risk", risk)
    novelty_value = _validate_real_number("novelty", novelty)
    uncertainty_value = _validate_real_number("uncertainty", uncertainty)
    information_gain_numeric = _validate_real_number(
        "information_gain_value",
        information_gain_value,
        allow_negative=True,
    )
    raw_score = (
        resolved["utility"] * utility
        + resolved["novelty"] * novelty_value
        + resolved["uncertainty"] * uncertainty_value
        + resolved["information_gain"] * information_gain_numeric
        - resolved["risk"] * risk_value
    )
    return max(0.0, raw_score)


class RealityMarketplace:
    """Multi-resource market clearing over compute, memory, and energy bids.

    The marketplace reuses `ComputeAllocator` for the proportional-normalization
    step once a Phase 7 bid score has been computed. It therefore extends the
    earlier static allocator rather than silently reimplementing its core
    budget-splitting behavior.
    """

    def __init__(self, *, allocation_weights: Mapping[str, float] | None = None) -> None:
        self.allocation_weights = _resolve_weights(allocation_weights)
        self._accounts: dict[str, RealityAccount] = {}
        self._active_bids: dict[str, MarketBid] = {}
        self._last_allocations: dict[str, dict[str, float]] = {}
        self._last_scores: dict[str, float] = {}
        self._round_index = 0
        self._priority_allocator = ComputeAllocator(
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
            delta=1.0,
            epsilon=_EPSILON,
        )
        self._share_allocator = ComputeAllocator(
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            delta=1.0,
            epsilon=_EPSILON,
        )

    @property
    def round_index(self) -> int:
        """Number of completed market-clearing rounds."""
        return self._round_index

    @property
    def accounts(self) -> dict[str, RealityAccount]:
        """Return a shallow copy of registered accounts by id."""
        return dict(self._accounts)

    @property
    def active_bids(self) -> list[MarketBid]:
        """Return the currently queued bids for the next round."""
        return list(self._active_bids.values())

    def register(self, account: RealityAccount) -> RealityAccount:
        """Register one market participant."""
        if not isinstance(account, RealityAccount):
            raise TypeError("account must be a RealityAccount")
        if account.reality_id in self._accounts:
            raise ValueError(f"reality {account.reality_id!r} is already registered")
        self._accounts[account.reality_id] = account
        self._last_allocations[account.reality_id] = {name: 0.0 for name in RESOURCE_NAMES}
        self._last_scores.setdefault(account.reality_id, 0.0)
        return account

    def submit_bid(self, bid: MarketBid) -> MarketBid:
        """Queue one bid for the next market-clearing round."""
        if not isinstance(bid, MarketBid):
            raise TypeError("bid must be a MarketBid")
        self._require_account(bid.reality_id)
        if bid.reality_id in self._active_bids:
            raise ValueError(
                f"reality {bid.reality_id!r} already has an active bid; "
                "submit one multi-resource bid per round"
            )
        self._active_bids[bid.reality_id] = bid
        return bid

    def score_bid(self, bid: MarketBid) -> float:
        """Return a bid's effective market score after account-level modifiers."""
        account = self._require_account(bid.reality_id)
        base_score = compute_allocation_score(
            account.expected_utility if bid.expected_utility is None else bid.expected_utility,
            account.risk if bid.risk is None else bid.risk,
            account.novelty if bid.novelty is None else bid.novelty,
            bid.uncertainty,
            bid.information_gain,
            weights=self.allocation_weights,
        )
        if base_score <= 0.0:
            return 0.0
        profile = RoomComputeProfile(
            permission=account.priority,
            uncertainty=base_score,
            risk=0.0,
            value=1.0 + account.historical_performance,
        )
        return float(self._priority_allocator.priority(profile))

    def clear_market(self, capacities: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Allocate each supplied resource pool across the queued bids.

        Each resource pool is cleared independently, but every resource uses the
        same per-bid market score. Allocations are proportional to the effective
        score, subject to:

        * the bid's requested amount,
        * the bid's optional per-resource `max_allocation`,
        * the account's per-resource budget.
        """
        validated_capacities = _validate_resource_amounts("capacities", capacities, allow_empty=True)
        round_allocations = {
            reality_id: {resource: 0.0 for resource in RESOURCE_NAMES}
            for reality_id in self._accounts
        }
        bid_scores = {
            reality_id: self.score_bid(bid)
            for reality_id, bid in self._active_bids.items()
        }

        for resource, total_capacity in validated_capacities.items():
            if total_capacity <= 0.0:
                continue
            resource_bids = [
                bid for bid in self._active_bids.values() if bid.requested_resources.get(resource, 0.0) > 0.0
            ]
            if not resource_bids:
                continue

            caps = {bid.reality_id: self._effective_cap(bid, resource) for bid in resource_bids}
            open_ids = {bid.reality_id for bid in resource_bids if caps[bid.reality_id] > 0.0}
            remaining = total_capacity

            while remaining > _EPSILON and open_ids:
                open_bids = [self._active_bids[reality_id] for reality_id in open_ids]
                open_scores = [bid_scores[bid.reality_id] for bid in open_bids]
                if sum(open_scores) <= _EPSILON:
                    proposed = np.full(len(open_bids), remaining / len(open_bids), dtype=float)
                else:
                    profiles = [
                        RoomComputeProfile(
                            permission=1.0,
                            uncertainty=1.0,
                            risk=0.0,
                            value=max(bid_scores[bid.reality_id], 0.0),
                        )
                        for bid in open_bids
                    ]
                    proposed = self._share_allocator.allocate(profiles, remaining)

                distributed = 0.0
                closed_ids: list[str] = []
                for bid, share in zip(open_bids, proposed, strict=True):
                    granted_so_far = round_allocations[bid.reality_id][resource]
                    headroom = caps[bid.reality_id] - granted_so_far
                    grant = min(float(share), headroom)
                    round_allocations[bid.reality_id][resource] += grant
                    distributed += grant
                    if round_allocations[bid.reality_id][resource] >= caps[bid.reality_id] - _EPSILON:
                        closed_ids.append(bid.reality_id)

                remaining = max(0.0, remaining - distributed)
                for reality_id in closed_ids:
                    open_ids.discard(reality_id)
                if distributed <= _EPSILON:
                    break

        self._last_allocations = copy.deepcopy(round_allocations)
        self._last_scores = {reality_id: bid_scores.get(reality_id, 0.0) for reality_id in self._accounts}
        self._active_bids.clear()
        self._round_index += 1
        return self.get_last_allocations()

    def update_historical_performance(
        self,
        realized_scores: Mapping[str, float],
        *,
        blend: float = 0.5,
    ) -> dict[str, float]:
        """Update account performance with an exponential moving average."""
        if not isinstance(realized_scores, Mapping):
            raise TypeError("realized_scores must be a mapping of reality id -> score")
        blend_value = _validate_real_number("blend", blend)
        if blend_value > 1.0:
            raise ValueError("blend must be in the closed interval [0, 1]")

        updated: dict[str, float] = {}
        for reality_id, realized in realized_scores.items():
            account = self._require_account(reality_id)
            realized_value = _validate_real_number(f"realized_scores[{reality_id!r}]", realized)
            account.historical_performance = (
                (1.0 - blend_value) * account.historical_performance
                + blend_value * realized_value
            )
            updated[reality_id] = account.historical_performance
        return updated

    def get_last_allocations(self) -> dict[str, dict[str, float]]:
        """Return the most recently cleared per-reality allocations."""
        return copy.deepcopy(self._last_allocations)

    def get_standings(self) -> list[tuple[str, float]]:
        """Return descending `(reality_id, effective_score)` standings."""
        return sorted(self._last_scores.items(), key=lambda item: (-item[1], item[0]))

    def _effective_cap(self, bid: MarketBid, resource: str) -> float:
        account = self._require_account(bid.reality_id)
        requested = bid.requested_resources.get(resource, 0.0)
        bid_cap = requested
        if bid.max_allocation is not None:
            bid_cap = min(bid_cap, bid.max_allocation.get(resource, requested))
        return min(requested, bid_cap, account.resource_budget(resource))

    def _require_account(self, reality_id: str) -> RealityAccount:
        account = self._accounts.get(reality_id)
        if account is None:
            raise KeyError(f"unknown reality id: {reality_id!r}")
        return account


__all__ = [
    "DEFAULT_SCORE_WEIGHTS",
    "MarketBid",
    "RESOURCE_NAMES",
    "RealityAccount",
    "RealityMarketplace",
    "compute_allocation_score",
]
