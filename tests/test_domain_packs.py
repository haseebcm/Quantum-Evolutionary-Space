import numpy as np
import pytest

import qes.domain_packs as domain_packs_module
from qes.domain_packs import (
    ControlPolicyPackConfig,
    DesignSpacePackConfig,
    DomainPackRegistry,
    ResourceAllocationPackConfig,
    ResourceJob,
    list_domain_packs,
    make_control_policy_space,
    make_design_space,
    make_resource_allocation_pack,
)
from qes.space import QESSpace


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


def test_control_policy_pack_builds_ready_space() -> None:
    pack = make_control_policy_space(
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
            branch_count=8,
            rng_seed=7,
        )
    )

    assert isinstance(pack.space, QESSpace)
    assert len(pack.space.active_rooms()) == 9


def test_control_policy_pack_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="target_state must have length 2"):
        make_control_policy_space(
            ControlPolicyPackConfig(
                state_dim=2,
                target_state=[1.0],
                state_lower=[-1.0, -1.0],
                state_upper=[1.0, 1.0],
            )
        )


def test_control_policy_pack_rejects_out_of_bounds_seed_gains() -> None:
    with pytest.raises(ValueError, match="initial_gains must lie within"):
        make_control_policy_space(
            ControlPolicyPackConfig(
                state_dim=1,
                target_state=[1.0],
                state_lower=[-1.0],
                state_upper=[1.0],
                gain_lower=[0.0, 0.0, 0.0],
                gain_upper=[1.0, 1.0, 1.0],
                initial_gains=[2.0, 0.0, 0.0],
            )
        )


def test_control_policy_pack_end_to_end_run_returns_sensible_policy() -> None:
    pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=1,
            target_state=[1.0],
            state_lower=[-2.0],
            state_upper=[2.0],
            initial_state=[0.0],
            controller_kind="pid",
            plant_decay=0.5,
            plant_gain=1.0,
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

    baseline = pack.evaluate_gains([0.8, 0.05, 0.01])
    result = pack.run(steps=10)

    assert result.objective <= baseline
    assert result.final_distance_to_target < 1.0
    assert result.active_rooms >= 1
    assert result.collapsed_rooms >= 0


def test_domain_pack_helpers_validate_shapes_ranges_and_zero_weights() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        domain_packs_module._as_1d_array("x", [[1.0], [2.0]])
    with pytest.raises(ValueError, match="must have length 2"):
        domain_packs_module._as_1d_array("x", [1.0], dim=2)
    np.testing.assert_allclose(domain_packs_module._as_scalar_or_vector("x", [1.0, 2.0], 2), [1.0, 2.0])
    with pytest.raises(ValueError, match="scalar or a length-2 vector"):
        domain_packs_module._as_scalar_or_vector("x", np.ones((2, 2)), 2)
    with pytest.raises(ValueError, match="must be finite"):
        domain_packs_module._as_scalar_or_vector("x", float("inf"), 2)
    with pytest.raises(ValueError, match="positive integer"):
        domain_packs_module._validate_positive_int("n", True)
    with pytest.raises(ValueError, match="finite value > 0"):
        domain_packs_module._validate_positive_float("alpha", 0.0)
    with pytest.raises(ValueError, match="must share the same shape"):
        domain_packs_module._validate_bounds("a", [0.0], "b", [1.0, 2.0])
    with pytest.raises(ValueError, match="must be <= b elementwise"):
        domain_packs_module._validate_bounds("a", [2.0], "b", [1.0])
    np.testing.assert_allclose(domain_packs_module._normalized_weights([0.0, 0.0]), [0.0, 0.0])


def test_control_policy_pack_validates_plant_and_controller_configuration() -> None:
    base = dict(
        state_dim=1,
        target_state=[1.0],
        state_lower=[-1.0],
        state_upper=[1.0],
    )

    with pytest.raises(ValueError, match="plant_decay must be >="):
        make_control_policy_space(ControlPolicyPackConfig(**base, plant_decay=-1.0))
    with pytest.raises(ValueError, match="plant_gain must be > 0"):
        make_control_policy_space(ControlPolicyPackConfig(**base, plant_gain=0.0))
    with pytest.raises(ValueError, match="controller_kind must be"):
        make_control_policy_space(ControlPolicyPackConfig(**base, controller_kind="bang-bang"))
    with pytest.raises(ValueError, match="max_workers must be >="):
        make_control_policy_space(ControlPolicyPackConfig(**base, max_workers=0))


def test_control_policy_pack_uses_default_gain_bounds_names_and_linear_control() -> None:
    linear_pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=2,
            target_state=[1.0, -1.0],
            state_lower=[-2.0, -2.0],
            state_upper=[2.0, 2.0],
            controller_kind="linear",
        )
    )
    pid_pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=2,
            target_state=[1.0, -1.0],
            state_lower=[-2.0, -2.0],
            state_upper=[2.0, 2.0],
            controller_kind="pid",
            gain_lower=[0.0] * 6,
        )
    )

    np.testing.assert_allclose(linear_pack._gain_space.reference, [1.0, 1.0])
    np.testing.assert_allclose(pid_pack._gain_space.reference, [1.0, 1.0, 0.1, 0.1, 0.01, 0.01])
    assert linear_pack._gain_names() == ["k_0", "k_1"]
    np.testing.assert_allclose(
        linear_pack._control_input(
            np.array([2.0, 3.0]),
            np.array([0.5, -1.0]),
            np.zeros(2),
            np.zeros(2),
        ),
        [1.0, -3.0],
    )


def test_control_policy_pack_rejects_partial_default_bounds_that_cross() -> None:
    with pytest.raises(ValueError, match="gain_lower must be <="):
        make_control_policy_space(
            ControlPolicyPackConfig(
                state_dim=1,
                target_state=[1.0],
                state_lower=[-1.0],
                state_upper=[1.0],
                gain_lower=[11.0, 6.0, 3.0],
                controller_kind="pid",
            )
        )


def test_control_policy_pack_best_room_errors_without_rooms_and_falls_back(monkeypatch) -> None:
    pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=1,
            target_state=[1.0],
            state_lower=[-2.0],
            state_upper=[2.0],
            controller_kind="linear",
        )
    )
    fallback_pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=1,
            target_state=[1.0],
            state_lower=[-2.0],
            state_upper=[2.0],
        )
    )
    for room in fallback_pack.space.rooms.values():
        room.state = "Collapsed"
    best_room, admitted = fallback_pack._best_room()

    assert admitted is False
    assert best_room.id in fallback_pack.space.rooms

    no_trace_pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=1,
            target_state=[1.0],
            state_lower=[-2.0],
            state_upper=[2.0],
        )
    )

    class NoTraceResult:
        kinds = {}

    monkeypatch.setattr(domain_packs_module.GenesisSelectionPipeline, "run", lambda self, rooms: NoTraceResult())
    fallback_room, admitted = no_trace_pack._best_room()
    assert admitted is False
    assert fallback_room.id in no_trace_pack.space.rooms

    pack.space.rooms.clear()
    with pytest.raises(RuntimeError, match="has no rooms"):
        pack._best_room()


def test_control_policy_pack_supports_upper_bound_only_defaults() -> None:
    pack = make_control_policy_space(
        ControlPolicyPackConfig(
            state_dim=1,
            target_state=[1.0],
            state_lower=[-1.0],
            state_upper=[1.0],
            gain_upper=[2.0, 3.0, 4.0],
        )
    )
    np.testing.assert_allclose(pack._gain_upper, [2.0, 3.0, 4.0])


def test_design_space_pack_builds_ready_space() -> None:
    pack = make_design_space(
        DesignSpacePackConfig(
            parameter_lower=[0.02, 0.02, 0.002],
            parameter_upper=[0.20, 0.20, 0.03],
            initial_design=[0.08, 0.12, 0.010],
            cost_fn=beam_mass,
            constraint_fn=lambda x: 2.0e-5 - beam_stiffness(x),
            branch_count=10,
            rng_seed=11,
        )
    )

    assert isinstance(pack.space, QESSpace)
    assert len(pack.space.active_rooms()) == 11


def test_design_space_pack_rejects_non_finite_bounds() -> None:
    with pytest.raises(ValueError, match="must contain only finite values"):
        make_design_space(
            DesignSpacePackConfig(
                parameter_lower=[0.0, 0.0, np.nan],
                parameter_upper=[1.0, 1.0, 1.0],
                cost_fn=beam_mass,
                constraint_fn=lambda x: 0.0,
            )
        )


def test_design_space_pack_rejects_initial_design_outside_bounds() -> None:
    with pytest.raises(ValueError, match="initial_design must lie within"):
        make_design_space(
            DesignSpacePackConfig(
                parameter_lower=[0.0, 0.0],
                parameter_upper=[1.0, 1.0],
                initial_design=[2.0, 0.5],
                cost_fn=lambda x: float(np.sum(x)),
                constraint_fn=lambda x: 0.0,
            )
        )


def test_design_space_pack_end_to_end_run_returns_feasible_design() -> None:
    pack = make_design_space(
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

    baseline = pack.objective([0.08, 0.12, 0.010])
    result = pack.run(steps=10)

    assert result.objective <= baseline
    assert result.feasible
    assert result.constraint_value <= 0.0


def test_design_space_pack_validates_callables_midpoint_and_limits() -> None:
    with pytest.raises(ValueError, match="cost_fn must be callable"):
        make_design_space(
            DesignSpacePackConfig(
                parameter_lower=[0.0],
                parameter_upper=[1.0],
                cost_fn=None,
                constraint_fn=lambda x: 0.0,
            )
        )
    with pytest.raises(ValueError, match="constraint_fn must be callable"):
        make_design_space(
            DesignSpacePackConfig(
                parameter_lower=[0.0],
                parameter_upper=[1.0],
                cost_fn=lambda x: float(np.sum(x)),
                constraint_fn=None,
            )
        )
    with pytest.raises(ValueError, match="max_workers must be >="):
        make_design_space(
            DesignSpacePackConfig(
                parameter_lower=[0.0],
                parameter_upper=[1.0],
                cost_fn=lambda x: float(np.sum(x)),
                constraint_fn=lambda x: 0.0,
                max_workers=0,
            )
        )

    pack = make_design_space(
        DesignSpacePackConfig(
            parameter_lower=[0.0, 2.0],
            parameter_upper=[2.0, 4.0],
            cost_fn=lambda x: float(np.sum(x)),
            constraint_fn=lambda x: 0.0,
        )
    )
    np.testing.assert_allclose(pack._initial_design, [1.0, 3.0])


def test_design_space_pack_rejects_non_finite_constraint_and_cost_values() -> None:
    constraint_pack = make_design_space(
        DesignSpacePackConfig(
            parameter_lower=[0.0],
            parameter_upper=[1.0],
            cost_fn=lambda x: 1.0,
            constraint_fn=lambda x: float("nan"),
        )
    )
    with pytest.raises(ValueError, match="constraint_fn must return a finite float"):
        constraint_pack.constraint_value([0.5])

    cost_pack = make_design_space(
        DesignSpacePackConfig(
            parameter_lower=[0.0],
            parameter_upper=[1.0],
            cost_fn=lambda x: float("nan"),
            constraint_fn=lambda x: 0.0,
        )
    )
    with pytest.raises(ValueError, match="cost_fn must return a finite float"):
        cost_pack.cost([0.5])


def test_design_space_pack_best_room_errors_without_rooms_and_can_fallback(monkeypatch) -> None:
    pack = make_design_space(
        DesignSpacePackConfig(
            parameter_lower=[0.0],
            parameter_upper=[1.0],
            cost_fn=lambda x: float(np.sum(x)),
            constraint_fn=lambda x: 0.0,
        )
    )
    pack.space.rooms.clear()
    with pytest.raises(RuntimeError, match="has no rooms"):
        pack._best_room()

    fallback_pack = make_design_space(
        DesignSpacePackConfig(
            parameter_lower=[0.0],
            parameter_upper=[1.0],
            cost_fn=lambda x: float(np.sum(x)),
            constraint_fn=lambda x: 0.0,
        )
    )

    class NoSurvivorResult:
        kinds = {"design": type("Trace", (), {"survivor": None})()}

    monkeypatch.setattr(domain_packs_module.GenesisSelectionPipeline, "run", lambda self, rooms: NoSurvivorResult())
    best_room, admitted = fallback_pack._best_room()
    assert admitted is False
    assert best_room in fallback_pack.space.rooms.values()


def test_resource_allocation_pack_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="total_budget"):
        make_resource_allocation_pack(
            ResourceAllocationPackConfig(
                jobs=[ResourceJob(name="job-a", demand=4.0, priority=2.0)],
                total_budget=-1.0,
            )
        )


def test_resource_allocation_pack_rejects_duplicate_job_names() -> None:
    jobs = [
        ResourceJob(name="job-a", demand=4.0, priority=2.0),
        ResourceJob(name="job-a", demand=3.0, priority=1.0),
    ]
    with pytest.raises(ValueError, match="job names must be unique"):
        make_resource_allocation_pack(ResourceAllocationPackConfig(jobs=jobs, total_budget=5.0))


def test_resource_allocation_pack_end_to_end_run_caps_by_demand_and_uses_budget() -> None:
    pack = make_resource_allocation_pack(
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

    result = pack.run()

    assert result.total_allocated == pytest.approx(8.0)
    assert result.allocations["critical"] >= result.allocations["explore"]
    assert result.unmet_demand["critical"] >= 0.0
    assert result.weighted_satisfaction > 0.0


@pytest.mark.parametrize(
    "job_kwargs",
    [
        {"name": "", "demand": 1.0, "priority": 1.0},
        {"name": "job", "demand": -1.0, "priority": 1.0},
        {"name": "job", "demand": 1.0, "priority": -1.0},
        {"name": "job", "demand": 1.0, "priority": 1.0, "permission": 1.5},
        {"name": "job", "demand": 1.0, "priority": 1.0, "uncertainty": -0.1},
        {"name": "job", "demand": 1.0, "priority": 1.0, "risk": -0.1},
    ],
)
def test_resource_job_rejects_invalid_fields(job_kwargs) -> None:
    with pytest.raises(ValueError):
        ResourceJob(**job_kwargs)


def test_resource_allocation_pack_validates_non_empty_jobs_and_min_share() -> None:
    with pytest.raises(ValueError, match="jobs must be non-empty"):
        make_resource_allocation_pack(ResourceAllocationPackConfig(jobs=[], total_budget=1.0))
    with pytest.raises(ValueError, match="min_share must be in"):
        make_resource_allocation_pack(
            ResourceAllocationPackConfig(
                jobs=[ResourceJob(name="job", demand=1.0, priority=1.0)],
                total_budget=1.0,
                min_share=1.5,
            )
        )


def test_resource_allocation_pack_handles_zero_demand_and_loop_breaks(monkeypatch) -> None:
    zero_demand_pack = make_resource_allocation_pack(
        ResourceAllocationPackConfig(
            jobs=[
                ResourceJob(name="job-a", demand=0.0, priority=1.0),
                ResourceJob(name="job-b", demand=0.0, priority=2.0),
            ],
            total_budget=5.0,
        )
    )
    zero_result = zero_demand_pack.run()
    assert zero_result.weighted_satisfaction == pytest.approx(1.0)
    assert zero_result.total_allocated == pytest.approx(0.0)

    zero_extra_pack = make_resource_allocation_pack(
        ResourceAllocationPackConfig(
            jobs=[
                ResourceJob(name="job-a", demand=1.0, priority=1.0),
                ResourceJob(name="job-b", demand=1.0, priority=2.0),
            ],
            total_budget=4.0,
        )
    )
    monkeypatch.setattr(zero_extra_pack._allocator, "allocate_with_floor", lambda profiles, total, min_share: np.zeros(2))
    monkeypatch.setattr(zero_extra_pack._allocator, "allocate", lambda profiles, total: np.zeros(len(profiles)))
    zero_extra_result = zero_extra_pack.run()
    assert zero_extra_result.total_allocated == pytest.approx(0.0)

    tiny_extra_pack = make_resource_allocation_pack(
        ResourceAllocationPackConfig(
            jobs=[
                ResourceJob(name="job-a", demand=1.0, priority=1.0),
                ResourceJob(name="job-b", demand=1.0, priority=2.0),
            ],
            total_budget=4.0,
        )
    )
    monkeypatch.setattr(tiny_extra_pack._allocator, "allocate_with_floor", lambda profiles, total, min_share: np.zeros(2))
    monkeypatch.setattr(
        tiny_extra_pack._allocator,
        "allocate",
        lambda profiles, total: np.full(len(profiles), 1e-20),
    )
    tiny_extra_result = tiny_extra_pack.run()
    assert tiny_extra_result.total_allocated == pytest.approx(0.0)

    updating_pack = make_resource_allocation_pack(
        ResourceAllocationPackConfig(
            jobs=[
                ResourceJob(name="job-a", demand=1.0, priority=1.0),
                ResourceJob(name="job-b", demand=1.0, priority=2.0),
            ],
            total_budget=1.0,
        )
    )
    calls = {"count": 0}

    def staged_allocate(profiles, total):
        calls["count"] += 1
        if calls["count"] == 1:
            return np.array([0.5, 0.25], dtype=float)
        return np.zeros(len(profiles), dtype=float)

    monkeypatch.setattr(updating_pack._allocator, "allocate_with_floor", lambda profiles, total, min_share: np.zeros(2))
    monkeypatch.setattr(updating_pack._allocator, "allocate", staged_allocate)
    updating_result = updating_pack.run()
    assert updating_result.total_allocated == pytest.approx(0.75)


def test_domain_pack_registry_lists_available_packs() -> None:
    registry = DomainPackRegistry()

    assert registry.names() == ("control_policy", "design_space", "resource_allocation")
    assert [descriptor.name for descriptor in list_domain_packs()] == list(registry.names())
    assert registry.get("design_space").builder == "make_design_space"


def test_domain_pack_registry_rejects_unknown_pack() -> None:
    with pytest.raises(KeyError, match="unknown domain pack"):
        DomainPackRegistry().get("unknown")
