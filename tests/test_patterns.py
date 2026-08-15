import pytest

from qes.patterns import Pattern, PatternMemory


def make_pattern(intent="design", phi=0.0, cci=0.0, margin=1.0):
    return Pattern(intent=intent, context={}, payload={"foo": "bar"}, phi=phi, cci=cci, margin=margin)


def test_store_and_generate_returns_stored_pattern():
    mem = PatternMemory()
    p = make_pattern()
    mem.store(p)
    result = mem.generate("design", {})
    assert result is p
    assert result.uses == 1


def test_generate_returns_none_when_no_patterns_for_intent():
    mem = PatternMemory()
    assert mem.generate("unknown", {}) is None


def test_generate_picks_best_by_phi_then_cci_then_margin():
    mem = PatternMemory()
    worse = make_pattern(phi=1.0, cci=1.0, margin=0.1)
    better = make_pattern(phi=0.0, cci=0.0, margin=0.9)
    mem.store(worse)
    mem.store(better)
    assert mem.generate("design", {}) is better


def test_improves_on_compares_all_three_axes():
    better = make_pattern(phi=0.0, cci=0.0, margin=1.0)
    worse = make_pattern(phi=1.0, cci=1.0, margin=0.0)
    assert better.improves_on(worse)
    assert not worse.improves_on(better)


def test_retire_dominated_removes_strictly_worse_patterns():
    mem = PatternMemory()
    better = make_pattern(phi=0.0, cci=0.0, margin=1.0)
    worse = make_pattern(phi=1.0, cci=1.0, margin=0.0)
    mem.store(better)
    mem.store(worse)
    mem.retire_dominated("design")
    remaining = mem.all_patterns("design")
    assert worse not in remaining
    assert better in remaining


def test_all_patterns_without_intent_returns_everything():
    mem = PatternMemory()
    p1 = make_pattern(intent="a")
    p2 = make_pattern(intent="b")
    mem.store(p1)
    mem.store(p2)
    ids = {p.id for p in mem.all_patterns()}
    assert ids == {p1.id, p2.id}


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"intent": 1}, TypeError, "intent must be a string"),
        ({"intent": ""}, ValueError, "intent must be non-empty"),
        ({"context": []}, TypeError, "context must be a mapping"),
        ({"phi": float("nan")}, ValueError, "phi must be finite"),
        ({"cci": float("inf")}, ValueError, "cci must be finite"),
        ({"margin": float("-inf")}, ValueError, "margin must be finite"),
        ({"uses": "1"}, TypeError, "uses must be an integer"),
        ({"uses": -1}, ValueError, "uses must be >= 0"),
        ({"id": 1}, TypeError, "id must be a string"),
        ({"id": ""}, ValueError, "id must be non-empty"),
    ],
)
def test_pattern_validates_constructor_inputs(
    kwargs: dict[str, object], error_type: type[Exception], match: str
):
    data = {
        "intent": "design",
        "context": {},
        "payload": {"foo": "bar"},
        "phi": 0.0,
        "cci": 0.0,
        "margin": 1.0,
        "uses": 0,
        "id": "p-1",
    }
    data.update(kwargs)
    with pytest.raises(error_type, match=match):
        Pattern(**data)


def test_improves_on_requires_pattern_argument():
    with pytest.raises(TypeError, match="other must be a Pattern"):
        make_pattern().improves_on("not-a-pattern")  # type: ignore[arg-type]


def test_store_replaces_existing_pattern_with_same_id():
    mem = PatternMemory()
    original = Pattern(intent="design", context={}, payload={"foo": "bar"}, phi=1.0, id="same-id")
    replacement = Pattern(intent="design", context={}, payload={"foo": "bar"}, phi=0.0, id="same-id")
    mem.store(original)
    mem.store(replacement)
    stored = mem.all_patterns("design")
    assert stored == [replacement]


def test_pattern_memory_validates_inputs():
    mem = PatternMemory()
    with pytest.raises(TypeError, match="pattern must be a Pattern"):
        mem.store("not-a-pattern")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="intent must be a string"):
        mem.generate(1, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="context must be a mapping"):
        mem.generate("design", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="intent must be a string"):
        mem.retire_dominated(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="intent must be a string"):
        mem.all_patterns(1)  # type: ignore[arg-type]


def test_retire_dominated_noops_for_single_pattern():
    mem = PatternMemory()
    pattern = make_pattern()
    mem.store(pattern)
    mem.retire_dominated("design")
    assert mem.all_patterns("design") == [pattern]
