"""Phase 21 demo: reusable classical QES domain packs."""
from __future__ import annotations

import sys
import time
import tracemalloc

import numpy as np

from qes.domain_packs import (
    ControlPolicyPackConfig,
    DesignSpacePackConfig,
    ResourceAllocationPackConfig,
    ResourceJob,
    list_domain_packs,
    make_control_policy_space,
    make_design_space,
    make_resource_allocation_pack,
)


def beam_stiffness(design: np.ndarray) -> float:
    width, height, thickness = design
    inner_width = max(width - 2.0 * thickness, 0.0)
    inner_height = max(height - 2.0 * thickness, 0.0)
    return (width * height**3 - inner_width * inner_height**3) / 12.0


def beam_mass(design: np.ndarray) -> float:
    width, height, thickness = design
    inner_width = max(width - 2.0 * thickness, 0.0)
    inner_height = max(height - 2.0 * thickness, 0.0)
    area = width * height - inner_width * inner_height
    return 2700.0 * area


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tracemalloc.start()
    started = time.perf_counter()

    print("QES PHASE 21: DOMAIN PACKS")
    print("=" * 60)
    print("Available packs:")
    for descriptor in list_domain_packs():
        print(f"  - {descriptor.name}: {descriptor.summary}")
    print()

    control_pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=1,
            target_state=[1.0],
            state_lower=[-2.0],
            state_upper=[2.0],
            initial_state=[0.0],
            controller_kind="pid",
            gain_lower=[0.0, 0.0, 0.0],
            gain_upper=[4.0, 1.0, 0.5],
            initial_gains=[0.8, 0.05, 0.01],
            simulation_steps=20,
            branch_count=12,
            branch_scale=0.2,
            mutation_scale=0.08,
            rng_seed=3,
        )
    )
    control_result = control_pack.run(steps=10)
    print("[control policy pack]")
    print(f"  iterations          : {control_result.iterations}")
    print(f"  wall time           : {control_result.wall_time_seconds:.6f} s")
    print(f"  best gains          : {np.array2string(control_result.best_gains, precision=4)}")
    print(f"  objective           : {control_result.objective:.6f}")
    print(f"  final distance      : {control_result.final_distance_to_target:.6f}")
    print()

    design_pack = make_design_space(
        DesignSpacePackConfig(
            parameter_lower=[0.02, 0.02, 0.002],
            parameter_upper=[0.20, 0.20, 0.03],
            initial_design=[0.08, 0.12, 0.010],
            cost_fn=beam_mass,
            constraint_fn=lambda x: 2.0e-5 - beam_stiffness(x),
            branch_count=18,
            mutation_scale=0.003,
            rng_seed=5,
        )
    )
    design_result = design_pack.run(steps=10)
    print("[design space pack]")
    print(f"  iterations          : {design_result.iterations}")
    print(f"  wall time           : {design_result.wall_time_seconds:.6f} s")
    print(f"  best design         : {np.array2string(design_result.best_design, precision=5)}")
    print(f"  cost                : {design_result.cost:.6f}")
    print(f"  constraint value    : {design_result.constraint_value:.8f} (<= 0 is feasible)")
    print()

    allocation_pack = make_resource_allocation_pack(
        ResourceAllocationPackConfig(
            jobs=[
                ResourceJob(
                    name="critical",
                    demand=5.0,
                    priority=4.0,
                    permission=0.95,
                    uncertainty=0.6,
                    risk=0.2,
                ),
                ResourceJob(
                    name="batch",
                    demand=3.0,
                    priority=2.0,
                    permission=0.90,
                    uncertainty=0.5,
                    risk=0.1,
                ),
                ResourceJob(
                    name="explore",
                    demand=2.0,
                    priority=1.0,
                    permission=0.85,
                    uncertainty=0.9,
                    risk=0.3,
                ),
            ],
            total_budget=8.0,
            min_share=0.2,
        )
    )
    allocation_result = allocation_pack.run()
    print("[resource allocation pack]")
    print(f"  wall time           : {allocation_result.wall_time_seconds:.6f} s")
    print(f"  total allocated     : {allocation_result.total_allocated:.6f}")
    print(f"  weighted satisfaction: {allocation_result.weighted_satisfaction:.6f}")
    for name, allocation in allocation_result.allocations.items():
        unmet = allocation_result.unmet_demand[name]
        print(f"  {name:18s} allocation={allocation:.4f}  unmet={unmet:.4f}")
    print()

    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (ordinary local CPU/RAM for classical search)")
    print(f"  wall time            : {elapsed:.6f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  Honest note: these packs wrap classical NumPy simulations, bounded")
    print("  local search, and priority-based allocation; they are not literal")
    print("  quantum computing and not production control/manufacturing software.")


if __name__ == "__main__":
    main()
