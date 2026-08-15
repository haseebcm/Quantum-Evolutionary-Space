import pytest

from qes.qsee import BIG11, QEL, QSEE11L, AcrosV12BIESync


def test_qel_default_name_from_index():
    qel = QEL(index=1)
    assert qel.name == "Logical Expansion"
    qel11 = QEL(index=11)
    assert qel11.name == "Meta-Evolution & Autonomous Self-Refinement"


def test_qel_preserves_explicit_name():
    qel = QEL(index=2, name="custom")
    assert qel.name == "custom"


def test_qel_evolve_records_delta_and_updates_state():
    qel = QEL(index=1, state=0)
    result = qel.evolve(lambda s: s + 1)
    assert result == 1
    assert qel.state == 1
    assert qel.deltas == [{"from": 0, "to": 1}]


def test_qel_index_out_of_range_rejected():
    with pytest.raises(ValueError):
        QEL(index=0)
    with pytest.raises(ValueError):
        QEL(index=12)


def test_big11_requires_exactly_eleven_qels():
    with pytest.raises(ValueError):
        BIG11(qels=[QEL(index=1)])


def test_big11_rejects_duplicate_or_out_of_range_indices():
    with pytest.raises(ValueError):
        BIG11(qels=[QEL(index=1)] * 11)


def test_big11_evolve_all_rejects_unknown_index():
    gate = BIG11()
    with pytest.raises(KeyError):
        gate.evolve_all({12: lambda s: "x"})


def test_big11_default_creates_eleven_qels():
    gate = BIG11()
    assert len(gate.qels) == 11
    assert [q.index for q in gate.qels] == list(range(1, 12))


def test_big11_evolve_all_only_touches_specified_streams():
    gate = BIG11()
    results = gate.evolve_all({1: lambda s: "a", 3: lambda s: "c"})
    assert results == {1: "a", 3: "c"}
    assert gate.qels[0].state == "a"
    assert gate.qels[2].state == "c"
    assert gate.qels[1].state is None  # untouched


def test_big11_merge_is_forbidden():
    gate = BIG11()
    with pytest.raises(PermissionError):
        gate.merge()


def test_big11_snapshot_states_is_isolated_copy():
    gate = BIG11()
    gate.evolve_all({1: lambda s: [1, 2, 3]})
    snapshot = gate.snapshot_states()
    snapshot[1].append(99)
    assert gate.qels[0].state == [1, 2, 3]


def test_sync_gate_aligns_sequentially_and_is_deterministic():
    gate = BIG11()
    gate.evolve_all({i: (lambda s, i=i: i * 10) for i in range(1, 12)})
    sync = AcrosV12BIESync()
    result = sync.synchronize(gate, combine_fn=lambda states: sum(states))
    assert result.aligned_states == [i * 10 for i in range(1, 12)]
    assert result.output == sum(i * 10 for i in range(1, 12))


def test_qsee11l_evolve_and_request_output():
    system = QSEE11L()
    system.evolve({1: lambda s: 5, 2: lambda s: 7})
    result = system.request_output(combine_fn=lambda states: states[0] + states[1])
    assert result.output == 12
    # streams continue evolving independently after output
    system.evolve({1: lambda s: s + 1})
    assert system.gate.qels[0].state == 6
