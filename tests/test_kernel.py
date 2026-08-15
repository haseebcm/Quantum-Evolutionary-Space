import numpy as np
import pytest

from qes.kernel import (
    ComputableGeometry,
    ExistenceKernel,
    RealitySelector,
    RecursiveEntity,
    RejectionError,
    SelfGeneratingDomainIntelligence,
)

# -- Layer 1/2: Null Origin & Existence Allowance --------------------------------


def test_null_origin_is_none():
    assert ExistenceKernel.null_origin(dim=3) is None


def test_allow_rejects_null_candidate():
    with pytest.raises(RejectionError):
        ExistenceKernel.allow(None)


def test_allow_instantiates_a_candidate():
    instantiated = ExistenceKernel.allow(np.array([1.0, 2.0]))
    assert np.allclose(instantiated, [1.0, 2.0])


# -- Layer 3: Construction --------------------------------------------------------


def test_construct_accumulates_within_bound():
    kernel = ExistenceKernel(delta_max=1.0)
    result = kernel.construct(complexity_t=2.0, delta=0.5)
    assert result.complexity == pytest.approx(2.5)
    assert result.delta == pytest.approx(0.5)


def test_construct_rejects_delta_outside_bound():
    kernel = ExistenceKernel(delta_max=1.0)
    with pytest.raises(RejectionError):
        kernel.construct(complexity_t=0.0, delta=1.5)
    with pytest.raises(RejectionError):
        kernel.construct(complexity_t=0.0, delta=0.0)
    with pytest.raises(RejectionError):
        kernel.construct(complexity_t=0.0, delta=-0.1)


# -- Layer 4: Mechanism (identity-preserving transition) -------------------------


def test_identity_preserving_within_epsilon():
    kernel = ExistenceKernel(epsilon=0.1)
    assert kernel.is_identity_preserving(np.array([0.0]), np.array([0.05]))
    assert not kernel.is_identity_preserving(np.array([0.0]), np.array([1.0]))


# -- Layer 5: Operation ------------------------------------------------------------


def test_operate_accepts_identity_preserving_transform():
    kernel = ExistenceKernel(epsilon=0.2)
    result = kernel.operate(np.array([1.0, 1.0]), lambda s: s + 0.1)
    assert result.identity_preserved
    assert np.allclose(result.state, [1.1, 1.1])


def test_operate_rejects_identity_breaking_transform():
    kernel = ExistenceKernel(epsilon=0.01)
    with pytest.raises(RejectionError):
        kernel.operate(np.array([1.0]), lambda s: s + 5.0)


# -- Layer 6: Rejection -------------------------------------------------------------


def test_reject_returns_state_when_admissible():
    state = np.array([0.5])
    result = ExistenceKernel.reject(state, admissible_fn=lambda s: bool(np.all(s >= 0)))
    assert result is not None
    assert np.allclose(result, state)


def test_reject_returns_none_when_inadmissible():
    result = ExistenceKernel.reject(np.array([-1.0]), admissible_fn=lambda s: bool(np.all(s >= 0)))
    assert result is None


# -- Layer 7: Persistence / Materialization -----------------------------------------


def test_find_persistent_state_converges_to_fixed_point():
    kernel = ExistenceKernel()
    # Contraction toward 3.0: O(s) = s + (3 - s) * 0.5
    result = kernel.find_persistent_state(
        state=np.array([0.0]), op_fn=lambda s: s + (3.0 - s) * 0.5, tolerance=1e-6
    )
    assert result.converged
    assert np.allclose(result.state, [3.0], atol=1e-4)


def test_find_persistent_state_reports_non_convergence():
    kernel = ExistenceKernel()
    # Oscillates forever, never settles.
    result = kernel.find_persistent_state(
        state=np.array([1.0]), op_fn=lambda s: -s, max_iterations=10, tolerance=1e-9
    )
    assert not result.converged
    assert result.iterations == 10


# -- Layer 8: Bounded Computational Energy -------------------------------------------


def test_bounded_energy_within_range():
    kernel = ExistenceKernel(energy_max=2.0)
    energy = kernel.bounded_energy(complexity_t=1.0, complexity_t1=2.0, dt=1.0)
    assert energy == pytest.approx(1.0)


def test_bounded_energy_rejects_out_of_range():
    kernel = ExistenceKernel(energy_max=1.0)
    with pytest.raises(RejectionError):
        kernel.bounded_energy(complexity_t=0.0, complexity_t1=5.0, dt=1.0)
    with pytest.raises(RejectionError):
        # zero/negative energy (no progress or regression) is also rejected
        kernel.bounded_energy(complexity_t=5.0, complexity_t1=5.0, dt=1.0)


def test_bounded_energy_requires_positive_dt():
    kernel = ExistenceKernel()
    with pytest.raises(ValueError):
        kernel.bounded_energy(complexity_t=0.0, complexity_t1=1.0, dt=0.0)


# -- Layer 9: Computable Geometry -----------------------------------------------------


def test_computable_geometry_from_grid_and_disjointness():
    def field(point: tuple[int, ...]) -> np.ndarray:
        x = point[0]
        return np.array([1.0]) if x < 5 else np.array([100.0])

    points = [(i,) for i in range(10)]
    geometry = ComputableGeometry.from_grid(
        field_fn=field, points=points, neighbor_offsets=[(1,)], epsilon=0.5
    )
    # Points 0-3 have identity-preserving neighbors within the same region;
    # point 4 straddles the jump at x=5 and is excluded.
    assert (0,) in geometry.support
    assert (4,) not in geometry.support


def test_computable_geometries_all_disjoint():
    g1 = ComputableGeometry(support=frozenset({(0,), (1,)}))
    g2 = ComputableGeometry(support=frozenset({(2,), (3,)}))
    g3 = ComputableGeometry(support=frozenset({(1,), (4,)}))
    assert ComputableGeometry.all_disjoint([g1, g2])
    assert not ComputableGeometry.all_disjoint([g1, g3])


# -- Layer 10: Recursive Autonomous Entities --------------------------------------------


def test_recursive_entity_step_within_bound():
    entity = RecursiveEntity(lambda_bound=0.5)
    result = entity.step(np.array([1.0]), lambda s: s + 0.1)
    assert result.within_bound
    assert result.divergence == pytest.approx(0.1)


def test_recursive_entity_step_exceeds_bound():
    entity = RecursiveEntity(lambda_bound=0.1)
    result = entity.step(np.array([1.0]), lambda s: s + 5.0)
    assert not result.within_bound


def test_recursive_entity_reproduces_at_fixed_point():
    entity = RecursiveEntity(lambda_bound=1.0)
    assert entity.reproduces(np.array([2.0]), lambda s: s, tolerance=1e-9)
    assert not entity.reproduces(np.array([2.0]), lambda s: s + 1.0, tolerance=1e-9)


def test_recursive_entity_requires_positive_lambda():
    with pytest.raises(ValueError):
        RecursiveEntity(lambda_bound=0.0)


# -- Layer 11: Self-Generating Domain Intelligence -----------------------------------------


def test_self_generating_domain_intelligence_preserves_invariant():
    generator = SelfGeneratingDomainIntelligence(admissible_fn=lambda d: bool(np.all(d >= 0)))
    result = generator.generate(
        domains=[np.array([1.0, 2.0]), np.array([0.5])], phi_fn=lambda d: d * 2
    )
    assert result.all_admissible
    assert len(result.domains) == 2


def test_self_generating_domain_intelligence_detects_invariant_violation():
    generator = SelfGeneratingDomainIntelligence(admissible_fn=lambda d: bool(np.all(d >= 0)))
    result = generator.generate(domains=[np.array([1.0])], phi_fn=lambda d: d - 10.0)
    assert not result.all_admissible


# -- Reality compiler: candidates vs realities ----------------------------------------------


def test_reality_selector_distinguishes_candidates_from_realities():
    # Phi maps everything toward the fixed point 0.0 except one true fixed point.
    selector = RealitySelector(phi_fn=lambda x: x * 0.0, tolerance=1e-6)
    result = selector.select([np.array([0.0]), np.array([5.0]), np.array([-3.0])])
    assert len(result.realities) == 1
    assert np.allclose(result.realities[0], [0.0])
    assert len(result.rejected) == 2


def test_reality_selector_is_reality_predicate():
    selector = RealitySelector(phi_fn=lambda x: x, tolerance=1e-9)
    # Identity operator: every candidate is already a fixed point / reality.
    assert selector.is_reality(np.array([42.0]))


def test_kernel_and_null_origin_validate_scalar_inputs():
    with pytest.raises(ValueError):
        ExistenceKernel(epsilon=float("inf"))
    with pytest.raises(ValueError):
        ExistenceKernel(epsilon=-1.0)
    with pytest.raises(ValueError):
        ExistenceKernel(delta_max=0.0)
    with pytest.raises(ValueError):
        ExistenceKernel(energy_max=0.0)
    with pytest.raises(TypeError):
        ExistenceKernel.null_origin(dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExistenceKernel.null_origin(dim=-1)
    with pytest.raises(ValueError):
        ExistenceKernel.allow(np.array([np.nan]))


def test_construct_mechanism_operation_and_reject_validate_inputs():
    kernel = ExistenceKernel()

    with pytest.raises(RejectionError):
        kernel.construct(complexity_t=-1.0, delta=0.5)
    with pytest.raises(ValueError, match="shape"):
        kernel.is_identity_preserving(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        kernel.operate(np.array([1.0]), "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ExistenceKernel.reject(np.array([1.0]), "bad")  # type: ignore[arg-type]


def test_persistence_and_energy_validate_bounds_and_iteration_controls():
    kernel = ExistenceKernel()

    with pytest.raises(TypeError):
        kernel.find_persistent_state(np.array([1.0]), "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        kernel.find_persistent_state(np.array([1.0]), lambda s: s, max_iterations=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        kernel.find_persistent_state(np.array([1.0]), lambda s: s, max_iterations=0)
    with pytest.raises(ValueError):
        kernel.find_persistent_state(np.array([1.0]), lambda s: s, tolerance=-1.0)
    with pytest.raises(ValueError):
        kernel.bounded_energy(complexity_t=-1.0, complexity_t1=1.0, dt=1.0)


def test_computable_geometry_validates_inputs_and_edge_cases():
    with pytest.raises(TypeError):
        ComputableGeometry.from_grid(123, [], [(0,)])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ComputableGeometry.from_grid(
            field_fn=lambda point: np.array([float(point[0])]),
            points=[(0,)],
            neighbor_offsets=[(0,)],
            epsilon=-1.0,
        )
    assert ComputableGeometry.from_grid(
        field_fn=lambda point: np.array([float(point[0])]),
        points=[],
        neighbor_offsets=[(0,)],
    ).support == frozenset()
    with pytest.raises(ValueError):
        ComputableGeometry.from_grid(
            field_fn=lambda point: np.array([0.0]),
            points=[(0,), (0, 0)],
            neighbor_offsets=[(0,)],
        )
    with pytest.raises(ValueError):
        ComputableGeometry.from_grid(
            field_fn=lambda point: np.array([0.0]),
            points=[(0,), (1,)],
            neighbor_offsets=[(0, 0)],
        )
    with pytest.raises(ValueError):
        ComputableGeometry.from_grid(
            field_fn=lambda point: np.array([0.0]) if point[0] == 0 else np.array([[1.0]]),
            points=[(0,), (1,)],
            neighbor_offsets=[(0,)],
        )
    with pytest.raises(TypeError):
        ComputableGeometry(support=frozenset()).overlaps(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ComputableGeometry.all_disjoint([object()])  # type: ignore[list-item]


def test_computable_geometry_overlaps_reports_intersection():
    left = ComputableGeometry(support=frozenset({(0,), (1,)}))
    right = ComputableGeometry(support=frozenset({(1,), (2,)}))
    assert left.overlaps(right)


def test_recursive_domain_and_reality_components_validate_callables_and_tolerances():
    entity = RecursiveEntity(lambda_bound=1.0)
    with pytest.raises(TypeError):
        entity.step(np.array([1.0]), "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        entity.reproduces(np.array([1.0]), "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        entity.reproduces(np.array([1.0]), lambda s: s, tolerance=-1.0)

    with pytest.raises(TypeError):
        SelfGeneratingDomainIntelligence("bad")  # type: ignore[arg-type]
    generator = SelfGeneratingDomainIntelligence(lambda _domain: True)
    with pytest.raises(TypeError):
        generator.generate([np.array([1.0])], "bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        RealitySelector(phi_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RealitySelector(phi_fn=lambda x: x, tolerance=-1.0)
