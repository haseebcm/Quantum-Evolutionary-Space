import numpy as np
import pytest

from qes.equation_forge import Equation, EquationForge
from qes.patterns import PatternMemory
from qes.permission import GenesisPermission
from qes.x_engine import (
    AcrosV12BIE,
    AcrosV13,
    ApexI,
    GeoM,
    x_engine_pipeline,
)


def test_v12_bie_hill_climbs_to_lower_score():
    forge = EquationForge(mutation_rate=0.2, rng=np.random.default_rng(0))
    seed = forge.seed(theta={"a": 10.0})
    v12 = AcrosV12BIE(forge, steps=25, scale=1.0)

    def score(eq):
        return abs(eq.theta["a"])

    stabilized = v12.stabilize(seed, score)
    assert score(stabilized) <= score(seed)


def test_v12_bie_rejects_negative_steps_and_scale():
    forge = EquationForge(rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="steps"):
        AcrosV12BIE(forge, steps=-1)
    with pytest.raises(ValueError, match="scale"):
        AcrosV12BIE(forge, scale=-0.1)


def test_v13_select_and_amplifying():
    v13 = AcrosV13(
        risk_fn=lambda c: c["risk"],
        inconsistency_fn=lambda c: c["inc"],
        instability_fn=lambda c: c["inst"],
    )
    candidates_round1 = [{"risk": 5, "inc": 1, "inst": 1}, {"risk": 1, "inc": 1, "inst": 1}]
    best1 = v13.select(candidates_round1)
    assert best1["risk"] == 1

    candidates_round2 = [{"risk": 0.1, "inc": 0.1, "inst": 0.1}]
    v13.select(candidates_round2)
    assert v13.amplifying() is True


def test_v13_select_empty_raises():
    v13 = AcrosV13(risk_fn=lambda c: 0, inconsistency_fn=lambda c: 0, instability_fn=lambda c: 0)
    with pytest.raises(ValueError):
        v13.select([])


def test_v13_amplifying_false_with_single_history():
    v13 = AcrosV13(risk_fn=lambda c: c, inconsistency_fn=lambda c: 0, instability_fn=lambda c: 0)
    v13.select([1, 2, 3])
    assert v13.amplifying() is False


def test_apex_i_executes_pipeline_and_traces():
    apex = ApexI()
    result = apex.execute([lambda x: x + 1, lambda x: x * 2], initial=1)
    assert result == 4
    assert apex.trace == [2, 4]


def test_apex_i_rejects_non_callable_stages():
    with pytest.raises(TypeError, match="callable"):
        ApexI().execute([lambda x: x, "bad-stage"], initial=1)  # type: ignore[list-item]


def test_geom_distance_and_project():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert GeoM.distance(a, b) == pytest.approx(5.0)
    weighted = GeoM.distance(a, b, w=np.array([0.0, 1.0]))
    assert weighted == pytest.approx(4.0)
    projected = GeoM.project(np.array([5.0, -5.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(projected, [1.0, -1.0])


def test_geom_centroid_and_bounding_radius():
    points = [np.array([0.0, 0.0]), np.array([2.0, 0.0])]
    centroid = GeoM.centroid(points)
    np.testing.assert_allclose(centroid, [1.0, 0.0])
    radius = GeoM.bounding_radius(points)
    assert radius == pytest.approx(1.0)


def test_geom_centroid_empty_raises():
    with pytest.raises(ValueError):
        GeoM.centroid([])


def test_geom_distance_rejects_shape_mismatches():
    with pytest.raises(ValueError, match="same shape"):
        GeoM.distance(np.array([0.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="must match"):
        GeoM.distance(np.array([0.0, 1.0]), np.array([0.0, 1.0]), w=np.array([1.0]))


def test_geom_bounding_radius_with_explicit_center():
    points = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    radius = GeoM.bounding_radius(points, center=np.array([0.0, 0.0]))
    assert radius == pytest.approx(4.0)


def test_x_engine_pipeline_rejects_when_not_admitted():
    permission = GenesisPermission(theta=0.01)
    forge = EquationForge(rng=np.random.default_rng(1))
    x = np.array([100.0, 100.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    result = x_engine_pipeline(
        intent="test",
        x=x,
        lower=lower,
        upper=upper,
        theta={"a": 1.0},
        permission=permission,
        forge=forge,
        score_fn=lambda eq: 0.0,
        executor_stages=[lambda v: v],
    )
    assert result.admitted is False
    assert result.equation is None


def test_x_engine_pipeline_full_flow_with_pattern_storage():
    permission = GenesisPermission(theta=10.0)
    forge = EquationForge(rng=np.random.default_rng(2))
    memory = PatternMemory()
    x = np.array([0.0, 0.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])

    result = x_engine_pipeline(
        intent="stabilize",
        x=x,
        lower=lower,
        upper=upper,
        theta={"a": 1.0},
        permission=permission,
        forge=forge,
        score_fn=lambda eq: abs(eq.theta["a"]),
        executor_stages=[lambda eq: eq.theta["a"] * 2],
        memory=memory,
    )
    assert result.admitted is True
    assert isinstance(result.equation, Equation)
    assert result.pattern is not None
    assert memory.all_patterns("stabilize") == [result.pattern]
    assert "distance_to_reference" in result.geometry


def test_x_engine_pipeline_without_memory_skips_pattern_storage():
    permission = GenesisPermission(theta=10.0)
    forge = EquationForge(rng=np.random.default_rng(3))
    x = np.array([0.0, 0.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])

    result = x_engine_pipeline(
        intent="no_memory",
        x=x,
        lower=lower,
        upper=upper,
        theta={"a": 1.0},
        permission=permission,
        forge=forge,
        score_fn=lambda eq: abs(eq.theta["a"]),
        executor_stages=[lambda eq: eq.theta["a"] * 2],
    )
    assert result.admitted is True
    assert result.pattern is None
