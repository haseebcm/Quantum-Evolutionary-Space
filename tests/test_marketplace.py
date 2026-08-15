from __future__ import annotations

import numpy as np
import pytest

from qes.marketplace import (
    DEFAULT_SCORE_WEIGHTS,
    MarketBid,
    RealityAccount,
    RealityMarketplace,
    compute_allocation_score,
)


def make_account(
    reality_id: str,
    *,
    compute_budget: float = 100.0,
    memory_budget: float = 50.0,
    energy_budget: float = 25.0,
    priority: float = 1.0,
    risk: float = 0.1,
    expected_utility: float = 1.0,
    novelty: float = 0.5,
    historical_performance: float = 1.0,
) -> RealityAccount:
    return RealityAccount(
        reality_id=reality_id,
        compute_budget=compute_budget,
        memory_budget=memory_budget,
        energy_budget=energy_budget,
        priority=priority,
        risk=risk,
        expected_utility=expected_utility,
        novelty=novelty,
        historical_performance=historical_performance,
    )


def make_bid(
    reality_id: str,
    *,
    compute: float = 0.0,
    memory: float = 0.0,
    energy: float = 0.0,
    uncertainty: float = 1.0,
    information_gain: float = 0.0,
    expected_utility: float | None = None,
    risk: float | None = None,
    novelty: float | None = None,
    max_allocation: dict[str, float] | None = None,
) -> MarketBid:
    requested = {
        name: amount
        for name, amount in {
            "compute": compute,
            "memory": memory,
            "energy": energy,
        }.items()
        if amount > 0.0
    }
    return MarketBid(
        reality_id=reality_id,
        requested_resources=requested,
        uncertainty=uncertainty,
        information_gain=information_gain,
        expected_utility=expected_utility,
        risk=risk,
        novelty=novelty,
        max_allocation=max_allocation,
    )


def test_reality_account_validates_and_exposes_resource_budget() -> None:
    account = make_account("alpha")

    assert account.resource_budget("compute") == pytest.approx(100.0)
    assert account.resource_budget("memory") == pytest.approx(50.0)
    assert account.resource_budget("energy") == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("reality_id", "", ValueError),
        ("compute_budget", -1.0, ValueError),
        ("memory_budget", float("nan"), ValueError),
        ("priority", True, TypeError),
    ],
)
def test_reality_account_rejects_invalid_inputs(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    kwargs: dict[str, object] = {
        "reality_id": "alpha",
        "compute_budget": 1.0,
        "memory_budget": 1.0,
        "energy_budget": 1.0,
    }
    kwargs[field] = value
    with pytest.raises(error_type):
        RealityAccount(**kwargs)  # type: ignore[arg-type]


def test_reality_account_rejects_unknown_resource_budget_lookup() -> None:
    with pytest.raises(ValueError):
        make_account("alpha").resource_budget("bandwidth")


def test_market_bid_validates_requested_resources_and_optional_caps() -> None:
    bid = MarketBid(
        reality_id="alpha",
        requested_resources={"compute": 10.0, "memory": 5.0},
        uncertainty=0.4,
        information_gain=-0.1,
        max_allocation={"compute": 8.0},
    )

    assert bid.requested_resources == {"compute": 10.0, "memory": 5.0}
    assert bid.max_allocation == {"compute": 8.0}


@pytest.mark.parametrize(
    ("requested_resources", "max_allocation", "error_type"),
    [
        ({}, None, ValueError),
        ({"compute": 0.0}, None, ValueError),
        ({"bandwidth": 1.0}, None, ValueError),
        ({"compute": 1.0}, {"memory": 1.0}, ValueError),
    ],
)
def test_market_bid_rejects_invalid_resource_shapes(
    requested_resources: dict[str, float],
    max_allocation: dict[str, float] | None,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        MarketBid(
            reality_id="alpha",
            requested_resources=requested_resources,
            uncertainty=1.0,
            max_allocation=max_allocation,
        )


def test_market_bid_from_entropy_change_reuses_phase8_information_gain_helper() -> None:
    bid = MarketBid.from_entropy_change(
        reality_id="alpha",
        requested_resources={"compute": 10.0},
        uncertainty_before=0.9,
        uncertainty_after=0.3,
        uncertainty=0.5,
    )

    assert bid.information_gain == pytest.approx(0.6)


def test_compute_allocation_score_matches_documented_default_formula() -> None:
    score = compute_allocation_score(2.0, 0.5, 1.0, 0.75, 1.5)
    expected = (
        DEFAULT_SCORE_WEIGHTS["utility"] * 2.0
        + DEFAULT_SCORE_WEIGHTS["novelty"] * 1.0
        + DEFAULT_SCORE_WEIGHTS["uncertainty"] * 0.75
        + DEFAULT_SCORE_WEIGHTS["information_gain"] * 1.5
        - DEFAULT_SCORE_WEIGHTS["risk"] * 0.5
    )
    assert score == pytest.approx(expected)


def test_compute_allocation_score_increases_with_utility() -> None:
    low = compute_allocation_score(0.5, 0.1, 0.2, 0.3, 0.4)
    high = compute_allocation_score(1.5, 0.1, 0.2, 0.3, 0.4)
    assert high > low


def test_compute_allocation_score_increases_with_information_gain() -> None:
    low = compute_allocation_score(1.0, 0.1, 0.2, 0.3, 0.0)
    high = compute_allocation_score(1.0, 0.1, 0.2, 0.3, 1.0)
    assert high > low


def test_compute_allocation_score_decreases_with_risk() -> None:
    low_risk = compute_allocation_score(1.0, 0.1, 0.2, 0.3, 0.4)
    high_risk = compute_allocation_score(1.0, 1.0, 0.2, 0.3, 0.4)
    assert high_risk < low_risk


def test_compute_allocation_score_is_floored_at_zero() -> None:
    score = compute_allocation_score(0.1, 10.0, 0.0, 0.0, -1.0)
    assert score == pytest.approx(0.0)


def test_compute_allocation_score_accepts_partial_weight_overrides() -> None:
    score = compute_allocation_score(
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        weights={"utility": 1.0, "risk": 0.0},
    )
    assert score == pytest.approx(1.0)


def test_compute_allocation_score_rejects_unknown_weight_key() -> None:
    with pytest.raises(KeyError):
        compute_allocation_score(1.0, 0.0, 0.0, 0.0, 0.0, weights={"surprise": 1.0})


def test_marketplace_registers_accounts_and_rejects_duplicates() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))

    with pytest.raises(ValueError):
        marketplace.register(make_account("alpha"))


def test_marketplace_accounts_property_returns_copy() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))

    accounts = marketplace.accounts
    accounts.clear()

    assert "alpha" in marketplace.accounts


def test_marketplace_submit_bid_requires_known_account() -> None:
    marketplace = RealityMarketplace()

    with pytest.raises(KeyError):
        marketplace.submit_bid(make_bid("missing", compute=5.0))


def test_marketplace_rejects_duplicate_active_bid_for_same_reality() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.submit_bid(make_bid("alpha", compute=5.0))

    with pytest.raises(ValueError):
        marketplace.submit_bid(make_bid("alpha", compute=1.0))


def test_marketplace_score_bid_uses_account_defaults_when_bid_omits_factors() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(
        make_account(
            "alpha",
            priority=2.0,
            risk=0.1,
            expected_utility=1.2,
            novelty=0.4,
            historical_performance=1.5,
        )
    )
    bid = make_bid("alpha", compute=10.0, uncertainty=0.7, information_gain=0.6)

    score = marketplace.score_bid(bid)

    assert score > 0.0


def test_marketplace_clear_market_returns_zero_allocations_when_no_bids_exist() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.register(make_account("beta"))

    allocations = marketplace.clear_market({"compute": 50.0})

    assert allocations["alpha"]["compute"] == pytest.approx(0.0)
    assert allocations["beta"]["compute"] == pytest.approx(0.0)


def test_marketplace_clear_market_allocates_more_to_higher_scoring_bid() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha", priority=1.0, historical_performance=1.0))
    marketplace.register(make_account("beta", priority=1.0, historical_performance=1.0))
    marketplace.submit_bid(
        make_bid(
            "alpha",
            compute=100.0,
            uncertainty=0.3,
            expected_utility=0.5,
            risk=0.5,
            novelty=0.1,
            information_gain=0.1,
        )
    )
    marketplace.submit_bid(
        make_bid(
            "beta",
            compute=100.0,
            uncertainty=1.0,
            expected_utility=2.0,
            risk=0.1,
            novelty=0.8,
            information_gain=1.0,
        )
    )

    allocations = marketplace.clear_market({"compute": 60.0})

    assert allocations["beta"]["compute"] > allocations["alpha"]["compute"]
    assert sum(item["compute"] for item in allocations.values()) == pytest.approx(60.0)


def test_marketplace_clear_market_respects_account_budget_caps() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha", compute_budget=10.0))
    marketplace.register(make_account("beta", compute_budget=100.0))
    marketplace.submit_bid(make_bid("alpha", compute=50.0, expected_utility=3.0, information_gain=1.0))
    marketplace.submit_bid(make_bid("beta", compute=50.0, expected_utility=1.0, information_gain=0.1))

    allocations = marketplace.clear_market({"compute": 40.0})

    assert allocations["alpha"]["compute"] == pytest.approx(10.0)
    assert allocations["beta"]["compute"] == pytest.approx(30.0)


def test_marketplace_clear_market_respects_bid_max_allocation_and_redistributes() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.register(make_account("beta"))
    marketplace.submit_bid(
        make_bid(
            "alpha",
            compute=100.0,
            expected_utility=5.0,
            uncertainty=1.0,
            information_gain=1.0,
            max_allocation={"compute": 12.0},
        )
    )
    marketplace.submit_bid(
        make_bid(
            "beta",
            compute=100.0,
            expected_utility=1.0,
            uncertainty=1.0,
            information_gain=0.2,
        )
    )

    allocations = marketplace.clear_market({"compute": 40.0})

    assert allocations["alpha"]["compute"] == pytest.approx(12.0)
    assert allocations["beta"]["compute"] == pytest.approx(28.0)


def test_marketplace_clear_market_supports_simultaneous_compute_memory_and_energy() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.register(make_account("beta"))
    marketplace.submit_bid(make_bid("alpha", compute=60.0, memory=20.0, energy=10.0, expected_utility=1.0))
    marketplace.submit_bid(make_bid("beta", compute=60.0, memory=20.0, energy=10.0, expected_utility=2.0))

    allocations = marketplace.clear_market({"compute": 30.0, "memory": 10.0, "energy": 6.0})

    assert sum(item["compute"] for item in allocations.values()) == pytest.approx(30.0)
    assert sum(item["memory"] for item in allocations.values()) == pytest.approx(10.0)
    assert sum(item["energy"] for item in allocations.values()) == pytest.approx(6.0)
    assert allocations["beta"]["compute"] > allocations["alpha"]["compute"]


def test_marketplace_equal_splits_when_all_scores_collapse_to_zero() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha", priority=1.0, historical_performance=0.0))
    marketplace.register(make_account("beta", priority=1.0, historical_performance=0.0))
    marketplace.submit_bid(
        make_bid(
            "alpha",
            compute=10.0,
            uncertainty=0.0,
            expected_utility=0.0,
            risk=10.0,
            novelty=0.0,
            information_gain=-1.0,
        )
    )
    marketplace.submit_bid(
        make_bid(
            "beta",
            compute=10.0,
            uncertainty=0.0,
            expected_utility=0.0,
            risk=9.0,
            novelty=0.0,
            information_gain=-1.0,
        )
    )

    allocations = marketplace.clear_market({"compute": 8.0})

    assert allocations["alpha"]["compute"] == pytest.approx(4.0)
    assert allocations["beta"]["compute"] == pytest.approx(4.0)


def test_marketplace_get_standings_returns_descending_scores() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.register(make_account("beta"))
    marketplace.submit_bid(make_bid("alpha", compute=20.0, expected_utility=1.0, information_gain=0.1))
    marketplace.submit_bid(make_bid("beta", compute=20.0, expected_utility=2.0, information_gain=1.0))
    marketplace.clear_market({"compute": 10.0})

    assert marketplace.get_standings()[0][0] == "beta"


def test_marketplace_update_historical_performance_uses_exponential_moving_average() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha", historical_performance=0.2))

    updated = marketplace.update_historical_performance({"alpha": 1.0}, blend=0.25)

    assert updated["alpha"] == pytest.approx(0.4)


def test_marketplace_historical_performance_changes_later_round_outcome() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha", priority=1.0, historical_performance=0.1))
    marketplace.register(make_account("beta", priority=1.0, historical_performance=0.1))

    marketplace.submit_bid(
        make_bid(
            "alpha",
            compute=50.0,
            uncertainty=0.8,
            expected_utility=1.0,
            risk=0.2,
            novelty=0.3,
            information_gain=0.4,
        )
    )
    marketplace.submit_bid(
        make_bid(
            "beta",
            compute=50.0,
            uncertainty=0.8,
            expected_utility=1.0,
            risk=0.2,
            novelty=0.3,
            information_gain=0.4,
        )
    )
    round_one = marketplace.clear_market({"compute": 20.0})
    assert round_one["alpha"]["compute"] == pytest.approx(10.0)
    assert round_one["beta"]["compute"] == pytest.approx(10.0)

    marketplace.update_historical_performance({"alpha": 2.0, "beta": 0.0}, blend=1.0)
    marketplace.submit_bid(
        make_bid(
            "alpha",
            compute=50.0,
            uncertainty=0.8,
            expected_utility=1.0,
            risk=0.2,
            novelty=0.3,
            information_gain=0.4,
        )
    )
    marketplace.submit_bid(
        make_bid(
            "beta",
            compute=50.0,
            uncertainty=0.8,
            expected_utility=1.0,
            risk=0.2,
            novelty=0.3,
            information_gain=0.4,
        )
    )
    round_two = marketplace.clear_market({"compute": 20.0})

    assert round_two["alpha"]["compute"] > round_two["beta"]["compute"]


def test_marketplace_update_historical_performance_rejects_unknown_account() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))

    with pytest.raises(KeyError):
        marketplace.update_historical_performance({"missing": 1.0})


@pytest.mark.parametrize("blend", [-0.1, 1.1])
def test_marketplace_update_historical_performance_rejects_out_of_range_blend(
    blend: float,
) -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))

    with pytest.raises(ValueError):
        marketplace.update_historical_performance({"alpha": 1.0}, blend=blend)


def test_marketplace_clear_market_clears_pending_bids_and_tracks_round_index() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.submit_bid(make_bid("alpha", compute=10.0))

    marketplace.clear_market({"compute": 5.0})

    assert marketplace.active_bids == []
    assert marketplace.round_index == 1


def test_reality_account_and_bid_reject_type_mismatches() -> None:
    with pytest.raises(TypeError):
        RealityAccount(
            reality_id=123,  # type: ignore[arg-type]
            compute_budget=1.0,
            memory_budget=1.0,
            energy_budget=1.0,
        )
    with pytest.raises(TypeError):
        RealityAccount(
            reality_id="alpha",
            compute_budget="bad",  # type: ignore[arg-type]
            memory_budget=1.0,
            energy_budget=1.0,
        )
    with pytest.raises(TypeError):
        RealityAccount(
            reality_id="alpha",
            compute_budget=1.0,
            memory_budget=1.0,
            energy_budget=1.0,
            metadata=[],
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MarketBid(
            reality_id="alpha",
            requested_resources={"compute": 1.0},
            uncertainty=1.0,
            metadata=[],
        )  # type: ignore[arg-type]


def test_market_bid_and_weights_validate_resource_names_and_mappings() -> None:
    with pytest.raises(TypeError):
        MarketBid(
            reality_id="alpha",
            requested_resources={1: 1.0},  # type: ignore[arg-type]
            uncertainty=1.0,
        )
    with pytest.raises(TypeError):
        MarketBid(
            reality_id="alpha",
            requested_resources=[],
            uncertainty=1.0,
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_allocation_score(1.0, 0.0, 0.0, 0.0, 0.0, weights=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_allocation_score(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            weights={name: 0.0 for name in DEFAULT_SCORE_WEIGHTS},
        )


def test_marketplace_register_submit_and_update_validate_argument_types() -> None:
    marketplace = RealityMarketplace()

    with pytest.raises(TypeError):
        marketplace.register("alpha")  # type: ignore[arg-type]

    marketplace.register(make_account("alpha"))

    with pytest.raises(TypeError):
        marketplace.submit_bid("alpha")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        marketplace.update_historical_performance(["alpha"])  # type: ignore[arg-type]


def test_marketplace_clear_market_skips_non_positive_capacity() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.submit_bid(make_bid("alpha", compute=10.0))

    allocations = marketplace.clear_market({"compute": 0.0})

    assert allocations["alpha"]["compute"] == pytest.approx(0.0)


def test_marketplace_clear_market_breaks_defensively_when_allocator_makes_no_progress() -> None:
    class ZeroAllocator:
        def allocate(self, profiles, remaining):
            del profiles, remaining
            return np.zeros(2, dtype=float)

    marketplace = RealityMarketplace()
    marketplace.register(make_account("alpha"))
    marketplace.register(make_account("beta"))
    marketplace.submit_bid(make_bid("alpha", compute=5.0, expected_utility=2.0))
    marketplace.submit_bid(make_bid("beta", compute=5.0, expected_utility=1.0))
    marketplace._share_allocator = ZeroAllocator()  # type: ignore[assignment]

    allocations = marketplace.clear_market({"compute": 4.0})

    assert allocations["alpha"]["compute"] == pytest.approx(0.0)
    assert allocations["beta"]["compute"] == pytest.approx(0.0)
