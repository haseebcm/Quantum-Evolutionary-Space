from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from qes.convergence import qes_entropy
from qes.information_gain import (
    InformationGainEstimator,
    allocate_by_information_gain,
    information_gain,
    population_entropy,
    rank_by_information_gain,
)
from qes.room import Room

information_gain_module = importlib.import_module("qes.information_gain")


def make_room(weight: float, offset: float = 0.0) -> Room:
    return Room(
        x=np.array([offset, offset + 0.1]),
        x_star=np.array([0.5, 0.5]),
        lower=-np.ones(2),
        upper=np.ones(2),
        activation=np.ones(2),
        weight=weight,
    )


def make_population(weights: list[float]) -> list[Room]:
    return [make_room(weight, offset=index * 0.01) for index, weight in enumerate(weights)]


def test_information_gain_matches_direct_entropy_difference() -> None:
    before_weights = [0.5, 0.3, 0.2]
    after_weights = [0.8, 0.1, 0.1]

    before_entropy = qes_entropy(before_weights)
    after_entropy = qes_entropy(after_weights)

    assert information_gain(before_entropy, after_entropy) == pytest.approx(
        before_entropy - after_entropy
    )


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("-inf"), float("nan")])
def test_information_gain_rejects_invalid_entropy_inputs(value: float) -> None:
    with pytest.raises(ValueError):
        information_gain(value, 0.1)

    with pytest.raises(ValueError):
        information_gain(0.1, value)


def test_information_gain_can_be_negative_when_uncertainty_increases() -> None:
    assert information_gain(0.2, 0.5) == pytest.approx(-0.3)


def test_population_entropy_matches_qes_entropy() -> None:
    rooms = make_population([0.55, 0.25, 0.2])

    assert population_entropy(rooms) == pytest.approx(qes_entropy([room.weight for room in rooms]))


def test_population_entropy_rejects_empty_room_list() -> None:
    with pytest.raises(ValueError):
        population_entropy([])


def test_estimate_realized_reports_positive_information_gain_for_convergence() -> None:
    estimator = InformationGainEstimator()
    rooms_before = make_population([0.4, 0.3, 0.3])
    rooms_after = make_population([0.85, 0.1, 0.05])

    score = estimator.estimate_realized(rooms_before, rooms_after)

    assert score > 0.0


def test_estimate_realized_reports_negative_information_gain_for_diffusion() -> None:
    estimator = InformationGainEstimator()
    rooms_before = make_population([0.9, 0.1, 0.0])
    rooms_after = make_population([1 / 3, 1 / 3, 1 / 3])

    score = estimator.estimate_realized(rooms_before, rooms_after)

    assert score < 0.0


def test_estimate_expected_returns_candidate_scores_that_rank_correctly() -> None:
    estimator = InformationGainEstimator()
    current = make_population([0.6, 0.25, 0.15])
    candidates = {
        "high_gain": (current, make_population([0.9, 0.07, 0.03])),
        "medium_gain": (current, make_population([0.7, 0.2, 0.1])),
        "negative_gain": (current, make_population([1 / 3, 1 / 3, 1 / 3])),
    }

    scores = estimator.estimate_expected(candidates)
    ranked_names = [name for name, _ in rank_by_information_gain(scores)]

    assert ranked_names == ["high_gain", "medium_gain", "negative_gain"]
    assert scores["high_gain"] > scores["medium_gain"] > scores["negative_gain"]


def test_rank_by_information_gain_sorts_descending_with_name_tie_break() -> None:
    ranked = rank_by_information_gain({"beta": 1.0, "alpha": 1.0, "gamma": 2.0})

    assert ranked == [("gamma", 2.0), ("alpha", 1.0), ("beta", 1.0)]


def test_allocate_by_information_gain_uses_floor_and_sums_to_budget() -> None:
    allocations = allocate_by_information_gain(
        {"focus": 3.0, "explore": 1.0, "discard": -2.0},
        total_budget=10.0,
        min_floor=1.0,
    )

    assert sum(allocations.values()) == pytest.approx(10.0)
    assert allocations["discard"] == pytest.approx(1.0)
    assert allocations["focus"] == pytest.approx(6.25)
    assert allocations["explore"] == pytest.approx(2.75)


def test_allocate_by_information_gain_splits_equally_when_all_scores_non_positive() -> None:
    allocations = allocate_by_information_gain(
        {"a": 0.0, "b": -1.0},
        total_budget=2.0,
    )

    assert allocations == pytest.approx({"a": 1.0, "b": 1.0})


@pytest.mark.parametrize(
    ("scores", "total_budget", "min_floor", "error_type"),
    [
        ({"a": 1.0}, -1.0, 0.0, ValueError),
        ({"a": 1.0}, 1.0, -0.1, ValueError),
        ({"a": 1.0, "b": 2.0}, 1.0, 0.6, ValueError),
        ({}, 1.0, 0.0, ValueError),
        ({"a": math.inf}, 1.0, 0.0, ValueError),
    ],
)
def test_allocate_by_information_gain_rejects_invalid_inputs(
    scores: dict[str, float],
    total_budget: float,
    min_floor: float,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        allocate_by_information_gain(scores, total_budget=total_budget, min_floor=min_floor)


def test_information_gain_and_candidate_names_reject_invalid_types() -> None:
    estimator = InformationGainEstimator()
    populations = (make_population([0.5, 0.5]), make_population([0.6, 0.4]))

    with pytest.raises(TypeError):
        information_gain(True, 0.1)
    with pytest.raises(TypeError):
        information_gain(object(), 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        estimator.estimate_expected({1: populations})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimator.estimate_expected({"": populations})


def test_population_entropy_rejects_invalid_room_sequences_and_weights() -> None:
    with pytest.raises(TypeError):
        population_entropy("not rooms")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        population_entropy([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        population_entropy([make_room(-0.1)])


def test_estimator_and_ranking_validate_input_shapes() -> None:
    estimator = InformationGainEstimator()

    with pytest.raises(TypeError):
        estimator.estimate_expected([])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        estimator.estimate_expected({"candidate": [make_population([1.0]), make_population([1.0])]})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        rank_by_information_gain([])  # type: ignore[arg-type]


def test_allocate_by_information_gain_handles_zero_candidates_and_exact_floor_budget() -> None:
    assert allocate_by_information_gain({}, total_budget=0.0) == {}
    assert allocate_by_information_gain(
        {"alpha": 1.0, "beta": 2.0},
        total_budget=1.0,
        min_floor=0.5,
    ) == pytest.approx({"alpha": 0.5, "beta": 0.5})


def test_allocate_by_information_gain_raises_if_final_sum_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(information_gain_module, "isclose", lambda *args, **kwargs: False)

    with pytest.raises(AssertionError, match="allocation does not sum"):
        allocate_by_information_gain({"alpha": 1.0}, total_budget=1.0)


def test_allocate_by_information_gain_validates_score_mapping_type() -> None:
    with pytest.raises(TypeError):
        allocate_by_information_gain([], total_budget=1.0)  # type: ignore[arg-type]


def test_allocate_by_information_gain_raises_if_exact_floor_sum_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_results = iter([True, False])
    monkeypatch.setattr(
        information_gain_module,
        "isclose",
        lambda *args, **kwargs: next(call_results),
    )

    with pytest.raises(AssertionError, match="allocation does not sum"):
        allocate_by_information_gain({"alpha": 1.0, "beta": 2.0}, total_budget=1.0, min_floor=0.5)
