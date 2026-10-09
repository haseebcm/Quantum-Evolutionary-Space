"""Independent basis-state checks for exported dense quantum operators."""
import itertools

import numpy as np
import pytest

from qes.quantum_compute import multi_qubit as mq
from qes.quantum_compute import quantum_gates as qg


@pytest.mark.parametrize('n', [2, 3, 4])
def test_cnot_every_ordered_pair_matches_bit_reference(n):
    for control, target in itertools.permutations(range(n), 2):
        matrix = qg.expand_two_qubit_gate(qg.cnot_matrix(), control, target, n)
        for index in range(1 << n):
            expected = index ^ (1 << (n - 1 - target)) if index & (1 << (n - 1 - control)) else index
            np.testing.assert_allclose(matrix[:, index], mq.basis_state(expected, n))


@pytest.mark.parametrize('n', [3, 4])
def test_toffoli_every_ordered_triple_matches_bit_reference(n):
    for a, b, target in itertools.permutations(range(n), 3):
        matrix = qg.expand_three_qubit_gate(qg.toffoli_matrix(), a, b, target, n)
        for index in range(1 << n):
            active = index & (1 << (n - 1 - a)) and index & (1 << (n - 1 - b))
            expected = index ^ (1 << (n - 1 - target)) if active else index
            np.testing.assert_allclose(matrix[:, index], mq.basis_state(expected, n))


@pytest.mark.parametrize('n', [2, 3, 4])
def test_partial_trace_matches_explicit_environment_sum(n):
    state = mq.random_state(n, np.random.default_rng(2026 + n))
    for count in range(n + 1):
        for keep in itertools.combinations(range(n), count):
            traced = [q for q in range(n) if q not in keep]
            expected = np.zeros((1 << count, 1 << count), dtype=complex)
            for env in itertools.product([0, 1], repeat=len(traced)):
                vector = []
                for retained in itertools.product([0, 1], repeat=count):
                    bits = dict(zip(keep, retained, strict=True)) | dict(zip(traced, env, strict=True))
                    index = sum(bits[q] << (n - 1 - q) for q in range(n))
                    vector.append(state[index])
                expected += np.outer(vector, np.conj(vector))
            np.testing.assert_allclose(mq.partial_trace(state, list(keep), n), expected, atol=1e-12)


@pytest.mark.parametrize('index', range(4))
def test_bell_reduced_state_and_entropy(index):
    state = mq.bell_state(index)
    np.testing.assert_allclose(mq.partial_trace(state, [0], 2), np.eye(2) / 2, atol=1e-12)
    assert mq.entanglement_entropy(state, [0], 2) == pytest.approx(1)
    assert mq.entanglement_entropy(mq.state_to_density_matrix(state), [0], 2) == pytest.approx(1)


@pytest.mark.parametrize('angle', [-np.pi, -0.1, 0, 0.7, np.pi])
def test_rotation_inverse_and_unitarity(angle):
    for make in (qg.Rx, qg.Ry, qg.Rz, qg.Phase):
        gate = make(angle)
        np.testing.assert_allclose(gate.conj().T @ gate, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(make(-angle) @ gate, np.eye(2), atol=1e-12)
