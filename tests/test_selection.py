import pytest

from qes.selection import GenesisSelection, GenesisSelectionPipeline, dominates, pareto_front


def test_admissible_requires_all_gates_up_to_layer():
    gates = [lambda x: x - 5, lambda x: x - 3]
    sel = GenesisSelection(gates, signature_fn=lambda x: "k", score_fn=lambda x: x)
    assert sel.admissible(4, layer=0)  # 4-5 <= 0
    assert not sel.admissible(4, layer=1)  # 4-3 > 0


def test_filter_layer_keeps_only_admissible_candidates():
    gates = [lambda x: x - 5]
    sel = GenesisSelection(gates, signature_fn=lambda x: "k", score_fn=lambda x: x)
    result = sel.filter_layer([1, 6, 3, 10], layer=0)
    assert result == [1, 3]


def test_classify_groups_by_signature():
    sel = GenesisSelection([], signature_fn=lambda x: x % 2, score_fn=lambda x: x)
    classes = sel.classify([1, 2, 3, 4, 5])
    assert classes[0] == [2, 4]
    assert classes[1] == [1, 3, 5]


def test_select_supreme_picks_lowest_score_per_kind():
    sel = GenesisSelection([], signature_fn=lambda x: x % 2, score_fn=lambda x: -x)
    survivors = sel.select_supreme([1, 2, 3, 4, 5])
    # score = -x, so argmin favors the largest x within each signature class
    assert survivors[0].candidate == 4
    assert survivors[1].candidate == 5


def test_run_applies_gates_then_selects_supreme():
    gates = [lambda x: x - 10]
    sel = GenesisSelection(gates, signature_fn=lambda x: x % 2, score_fn=lambda x: x)
    survivors = sel.run([1, 2, 3, 4, 5, 20])
    # 20 is excluded by the gate; among remaining, min score per parity class
    assert survivors[0].candidate == 2
    assert survivors[1].candidate == 1


def test_run_returns_empty_when_all_candidates_rejected():
    gates = [lambda x: x - 0]
    sel = GenesisSelection(gates, signature_fn=lambda x: x, score_fn=lambda x: x)
    survivors = sel.run([1, 2, 3])
    assert survivors == {}


def test_dominates_requires_no_worse_and_one_strictly_better():
    assert dominates((1.0, 2.0), (1.0, 3.0))
    assert not dominates((1.0, 2.0), (1.0, 2.0))
    assert not dominates((2.0, 1.0), (1.0, 2.0))


def test_pareto_front_keeps_only_non_dominated_candidates():
    candidates = [(1.0, 5.0), (2.0, 4.0), (5.0, 5.0), (1.0, 1.0)]
    front = pareto_front(candidates, objectives_fn=lambda c: c)
    # (1.0, 1.0) dominates everything else here (no worse and strictly better
    # in at least one objective vs. every other candidate).
    assert front == [(1.0, 1.0)]


def test_pareto_front_keeps_multiple_true_trade_offs():
    # Neither candidate dominates the other: first is better on objective 0,
    # second is better on objective 1.
    candidates = [(1.0, 5.0), (5.0, 1.0), (3.0, 3.0)]
    front = pareto_front(candidates, objectives_fn=lambda c: c)
    assert set(front) == {(1.0, 5.0), (5.0, 1.0), (3.0, 3.0)}


def test_select_pareto_per_kind_groups_then_filters():
    sel = GenesisSelection([], signature_fn=lambda c: c[0] % 2, score_fn=lambda c: c[1])
    candidates = [(0, 1.0, 5.0), (0, 5.0, 5.0), (1, 1.0, 1.0)]
    fronts = sel.select_pareto_per_kind(candidates, objectives_fn=lambda c: c[1:])
    assert (0, 5.0, 5.0) not in fronts[0]
    assert (0, 1.0, 5.0) in fronts[0]
    assert fronts[1] == [(1, 1.0, 1.0)]


def test_pipeline_tracks_late_new_kinds_and_eliminated_kinds():
    pipeline = GenesisSelectionPipeline(
        gates=[
            lambda x: 0 if x <= 2 else 1,
            lambda x: 0 if x <= 4 else 1,
            lambda x: 0 if x <= 4 else 1,
        ],
        signature_fn=lambda x: x,
        score_fn=lambda x: -x,
    )

    result = pipeline.run(
        [1, 3],
        generators=[
            lambda population: population,
            lambda population: list(population) + [4],
            lambda population: population,
        ],
    )

    assert result.kinds[1].survivor == 1
    assert result.kinds[3].eliminated_at_layer == 1
    assert result.kinds[4].first_seen_layer == 2
    assert result.population_history == [1, 2, 2]


def test_pipeline_requires_gates_and_handles_generator_free_runs():
    with pytest.raises(ValueError, match="non-empty"):
        GenesisSelectionPipeline([], signature_fn=lambda x: x, score_fn=lambda x: x)

    pipeline = GenesisSelectionPipeline(
        gates=[lambda x: x - 1, lambda x: x - 1],
        signature_fn=lambda x: x,
        score_fn=lambda x: x,
    )

    empty_result = pipeline.run([2, 3])
    assert empty_result.population_history == [0]
    assert empty_result.kinds[2].eliminated_at_layer == 1

    surviving_result = pipeline.run([0, 1])
    assert surviving_result.population_history == [2, 2]
    assert surviving_result.kinds[0].survivor == 0
