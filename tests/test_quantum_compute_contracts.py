import itertools
import json
import random
import zlib

import numpy as np
import pytest

from qes.quantum_compute import Circuit, ComputationalQubit
from qes.quantum_compute.quantum_gates import X
from qes.quantum_compute.serialize import (
    apply_diff,
    compute_diff,
    from_bytes,
    from_compressed,
    from_json,
    from_jsonl,
    to_bytes,
    to_compressed,
    to_json,
    to_jsonl,
)


def test_physical_diagnostics_normalize_computational_vectors():
    qubit = ComputationalQubit(primary=np.array([2.0, 0.0]))
    assert qubit.fidelity(qubit) == pytest.approx(1)
    assert qubit.purity() == pytest.approx(1)
    assert qubit.von_neumann_entropy() == pytest.approx(0)
    np.testing.assert_array_equal(qubit.primary, [2, 0])
    phase = ComputationalQubit(primary=np.array([2j, 0]))
    assert qubit.fidelity(phase) == pytest.approx(1)
    assert qubit.trace_distance(phase) == pytest.approx(0)
    assert qubit.bures_distance(phase) == pytest.approx(0)


@pytest.mark.parametrize("state", [[0, 0], [np.nan, 0], [np.inf, 0], [[1], [0]], []])
def test_invalid_quantum_vectors_are_rejected(state):
    with pytest.raises(ValueError):
        ComputationalQubit(primary=np.array(state)).density_matrix()


def test_density_constructor_supports_pure_states_only():
    state = np.array([1, 1j]) / np.sqrt(2)
    qubit = ComputationalQubit.from_density_matrix(np.outer(state, state.conj()))
    assert qubit.fidelity(ComputationalQubit(primary=state)) == pytest.approx(1)
    for invalid in [np.eye(2) / 2, np.eye(2), np.array([[1, 1], [0, 0]])]:
        with pytest.raises(ValueError):
            ComputationalQubit.from_density_matrix(invalid)


def test_bell_measurements_and_inverse():
    circuit = Circuit(2)
    circuit.apply_h(0)
    circuit.apply_cnot(0, 1)
    state = circuit.run()
    np.testing.assert_allclose(state, np.array([1, 0, 0, 1]) / np.sqrt(2))
    assert circuit.concurrence(state) == pytest.approx(1)
    shots = circuit.measure(2000, random.Random(10))
    assert set(shots) == {"00", "11"}
    assert 850 <= shots["00"] <= 1150
    np.testing.assert_allclose(circuit.inverse().run(state), [1, 0, 0, 0], atol=1e-14)


@pytest.mark.parametrize("control,target", list(itertools.permutations(range(3), 2)))
def test_local_cnot_matches_independent_bit_mapping(control, target):
    circuit = Circuit(3)
    circuit.apply_cnot(control, target)
    for index in range(8):
        state = np.eye(8, dtype=complex)[index]
        expected_index = index ^ (1 << (2 - target)) if index & (1 << (2 - control)) else index
        np.testing.assert_array_equal(circuit.run(state), np.eye(8)[expected_index])


@pytest.mark.parametrize("a,b,target", list(itertools.permutations(range(3), 3)))
def test_local_toffoli_matches_independent_bit_mapping(a, b, target):
    circuit = Circuit(3)
    circuit.apply_toffoli(a, b, target)
    for index in range(8):
        active = bool(index & (1 << (2 - a))) and bool(index & (1 << (2 - b)))
        expected = index ^ (1 << (2 - target)) if active else index
        np.testing.assert_array_equal(circuit.run(np.eye(8)[index]), np.eye(8)[expected])


def test_local_gate_storage_and_memory_limit():
    circuit = Circuit(14)
    circuit.apply_h(13)
    assert circuit.ops[0].matrix.shape == (2, 2)
    assert np.linalg.norm(circuit.run()) == pytest.approx(1)
    with pytest.raises(ValueError, match="memory limit"):
        Circuit(1000000)
    with pytest.raises(ValueError):
        Circuit(0)
    with pytest.raises(ValueError):
        Circuit(1).run(np.array([[1], [0]]))
    with pytest.raises(ValueError):
        Circuit(2).apply_controlled_u(0, 1, np.zeros((2, 2)))


def test_controlled_unitary_and_swap():
    circuit = Circuit(3)
    circuit.apply_x(2)
    circuit.apply_controlled_u(2, 0, X)
    circuit.apply_swap(0, 1)
    np.testing.assert_array_equal(circuit.run(), np.eye(8)[3])


@pytest.mark.parametrize("encode,decode", [(to_json, from_json), (to_bytes, from_bytes),
                                         (to_compressed, from_compressed)])
def test_safe_serialization_roundtrips_complex_null_and_reserved_keys(encode, decode):
    data = {"array": np.array([1j, 2 + 3j]), "null": None, "scalar": np.int64(3),
            "reserved": {"__qes_complex__": [1, 2]}, "_version": 99, "data": 7}
    restored = decode(encode(data))
    assert restored == {"array": [1j, 2 + 3j], "null": None, "scalar": 3,
                        "reserved": {"__qes_complex__": [1, 2]}, "_version": 99, "data": 7}


def test_jsonl_roundtrip_and_null_patch(tmp_path):
    path = str(tmp_path / "states.jsonl")
    to_jsonl([{"state": np.array([1j])}, None], path)
    assert from_jsonl(path) == [{"state": [1j]}, None]
    old, new = {"x": 1, "remove": None}, {"x": None, "y": 2}
    assert apply_diff(old, compute_diff(old, new)) == new
    with pytest.raises(ValueError):
        apply_diff(old, {"x": None})


def test_serialization_rejects_unknown_versions_nonfinite_and_pickle():
    with pytest.raises(ValueError, match="version"):
        from_json(json.dumps({"_version": 999, "data": {}}))
    with pytest.raises(ValueError):
        to_json(np.nan)
    with pytest.raises(ValueError):
        from_json('{"data": NaN}')
    # A recognizable legacy pickle header is rejected as data, never executed.
    with pytest.raises((ValueError, UnicodeDecodeError)):
        from_bytes(zlib.compress(b"\x80\x04N."))
    with pytest.raises(ValueError):
        from_bytes(to_bytes({})[:-2])


def test_decompression_limit(monkeypatch):
    import qes.quantum_compute.serialize as serialization

    encoded = to_bytes("x" * 1000)
    monkeypatch.setattr(serialization, "MAX_DECOMPRESSED_BYTES", 100)
    with pytest.raises(ValueError, match="size limit"):
        from_bytes(encoded)


def test_layer_checkpoint_restores_vectors_and_isolates_history():
    from qes.quantum_compute import Layer

    layer = Layer("checkpoint")
    layer.qubit.primary = np.array([1.0, 2.0])
    layer.history.append({"vector": np.array([1.0, 2.0])})
    checkpoint = layer.checkpoint()
    layer.qubit.primary[:] = 7
    layer.history[0]["vector"][:] = 8
    layer.restore(checkpoint)
    np.testing.assert_array_equal(layer.qubit.primary, [1, 2])
    np.testing.assert_array_equal(layer.history[0]["vector"], [1, 2])
    layer.history[0]["vector"][:] = 9
    np.testing.assert_array_equal(checkpoint["history"][0]["vector"], [1, 2])
