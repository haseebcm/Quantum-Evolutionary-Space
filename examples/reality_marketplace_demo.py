"""Phase 7 demo: classical resource-economy bidding inside QES.

This walkthrough registers a few reality accounts, lets them bid for compute,
memory, and energy with utility/risk/novelty/uncertainty/information-gain
signals, clears the market, then updates historical performance and runs a
second round. Everything here is ordinary Python arithmetic over dataclasses
and dictionaries, not a literal economy and not quantum computing.
"""
from __future__ import annotations

import time
import tracemalloc

from qes.marketplace import MarketBid, RealityAccount, RealityMarketplace


def print_round(
    title: str,
    marketplace: RealityMarketplace,
    allocations: dict[str, dict[str, float]],
) -> None:
    print(title)
    print("-" * 78)
    print(
        "reality        | score       | compute    | memory     | energy"
    )
    print("-" * 78)
    score_map = dict(marketplace.get_standings())
    for reality_id in sorted(allocations):
        allocation = allocations[reality_id]
        print(
            f"{reality_id:<14} | {score_map.get(reality_id, 0.0):>11.4f} | "
            f"{allocation['compute']:>10.2f} | {allocation['memory']:>10.2f} | "
            f"{allocation['energy']:>8.2f}"
        )


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 7: REALITY MARKETPLACE / RESOURCE ECONOMY")
    print("=" * 78)

    marketplace = RealityMarketplace()
    marketplace.register(
        RealityAccount(
            reality_id="stability_branch",
            compute_budget=500.0,
            memory_budget=160.0,
            energy_budget=90.0,
            priority=1.2,
            risk=0.15,
            expected_utility=1.1,
            novelty=0.35,
            historical_performance=0.9,
        )
    )
    marketplace.register(
        RealityAccount(
            reality_id="exploration_branch",
            compute_budget=700.0,
            memory_budget=180.0,
            energy_budget=120.0,
            priority=1.0,
            risk=0.35,
            expected_utility=1.5,
            novelty=0.95,
            historical_performance=0.8,
        )
    )
    marketplace.register(
        RealityAccount(
            reality_id="repair_branch",
            compute_budget=450.0,
            memory_budget=120.0,
            energy_budget=80.0,
            priority=1.4,
            risk=0.10,
            expected_utility=1.2,
            novelty=0.20,
            historical_performance=1.1,
        )
    )
    marketplace.register(
        RealityAccount(
            reality_id="speculative_branch",
            compute_budget=650.0,
            memory_budget=150.0,
            energy_budget=110.0,
            priority=0.9,
            risk=0.55,
            expected_utility=1.8,
            novelty=1.20,
            historical_performance=0.5,
        )
    )

    marketplace.submit_bid(
        MarketBid.from_entropy_change(
            reality_id="stability_branch",
            requested_resources={"compute": 350.0, "memory": 80.0, "energy": 45.0},
            uncertainty_before=0.95,
            uncertainty_after=0.60,
            uncertainty=0.55,
        )
    )
    marketplace.submit_bid(
        MarketBid.from_entropy_change(
            reality_id="exploration_branch",
            requested_resources={"compute": 500.0, "memory": 120.0, "energy": 60.0},
            uncertainty_before=1.10,
            uncertainty_after=0.45,
            uncertainty=0.90,
        )
    )
    marketplace.submit_bid(
        MarketBid.from_entropy_change(
            reality_id="repair_branch",
            requested_resources={"compute": 280.0, "memory": 70.0, "energy": 40.0},
            uncertainty_before=0.70,
            uncertainty_after=0.40,
            uncertainty=0.45,
        )
    )
    marketplace.submit_bid(
        MarketBid.from_entropy_change(
            reality_id="speculative_branch",
            requested_resources={"compute": 600.0, "memory": 130.0, "energy": 85.0},
            uncertainty_before=1.20,
            uncertainty_after=0.80,
            uncertainty=1.00,
            max_allocation={"compute": 420.0},
        )
    )

    round_one = marketplace.clear_market({"compute": 1000.0, "memory": 260.0, "energy": 160.0})
    print_round("ROUND 1: bids cleared from expected utility / novelty / IG signals", marketplace, round_one)

    updated = marketplace.update_historical_performance(
        {
            "stability_branch": 0.95,
            "exploration_branch": 1.40,
            "repair_branch": 1.15,
            "speculative_branch": 0.55,
        },
        blend=0.5,
    )
    print("\nUpdated historical performance after realized outcomes:")
    for reality_id in sorted(updated):
        print(f"  {reality_id:<20} -> {updated[reality_id]:.4f}")

    marketplace.submit_bid(
        MarketBid(
            reality_id="stability_branch",
            requested_resources={"compute": 320.0, "memory": 70.0, "energy": 40.0},
            uncertainty=0.45,
            information_gain=0.20,
        )
    )
    marketplace.submit_bid(
        MarketBid(
            reality_id="exploration_branch",
            requested_resources={"compute": 520.0, "memory": 130.0, "energy": 65.0},
            uncertainty=0.95,
            information_gain=0.75,
        )
    )
    marketplace.submit_bid(
        MarketBid(
            reality_id="repair_branch",
            requested_resources={"compute": 260.0, "memory": 65.0, "energy": 35.0},
            uncertainty=0.35,
            information_gain=0.18,
        )
    )
    marketplace.submit_bid(
        MarketBid(
            reality_id="speculative_branch",
            requested_resources={"compute": 560.0, "memory": 115.0, "energy": 80.0},
            uncertainty=0.85,
            information_gain=0.30,
            max_allocation={"compute": 350.0},
        )
    )

    round_two = marketplace.clear_market({"compute": 1000.0, "memory": 260.0, "energy": 160.0})
    print()
    print_round("ROUND 2: historical performance now tilts otherwise similar bids", marketplace, round_two)

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 78)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.6f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  objects competing    : 4 RealityAccount objects + 8 MarketBid objects")
    print(
        "  This demo is classical arithmetic over Python dataclasses: a weighted "
        "resource-allocation heuristic, not a literal marketplace and not "
        "quantum hardware."
    )


if __name__ == "__main__":
    main()
