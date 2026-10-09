import numpy as np
import pytest

from qes.convergence import (
    convergence_coefficient,
    gini_coefficient,
    kl_divergence,
    normalized_entropy,
    qes_entropy,
)


def test_entropy_zero_when_single_weight():
    assert qes_entropy([1.0]) == 0.0


def test_entropy_positive_for_uniform_distribution():
    weights = [0.25, 0.25, 0.25, 0.25]
    assert qes_entropy(weights) == pytest.approx(np.log(4))


def test_normalized_entropy_is_one_for_uniform_distribution():
    weights = [0.25, 0.25, 0.25, 0.25]
    assert normalized_entropy(weights) == pytest.approx(1.0)


def test_entropy_zero_when_all_weights_non_positive():
    assert qes_entropy([0.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        qes_entropy([0.0, 0.0, -1.0])


def test_normalized_entropy_is_zero_for_single_weight():
    assert normalized_entropy([1.0]) == 0.0


def test_convergence_coefficient_high_when_concentrated():
    concentrated = [0.97, 0.01, 0.01, 0.01]
    dispersed = [0.25, 0.25, 0.25, 0.25]
    assert convergence_coefficient(concentrated) > convergence_coefficient(dispersed)


def test_convergence_coefficient_bounds():
    weights = [0.5, 0.5]
    c = convergence_coefficient(weights)
    assert 0.0 <= c <= 1.0


def test_kl_divergence_zero_for_identical_distributions():
    weights = [0.25, 0.25, 0.25, 0.25]
    assert kl_divergence(weights, weights) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_positive_when_distributions_differ():
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    assert kl_divergence(p, q) > 0.0


def test_kl_divergence_raises_on_mismatched_length():
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [1.0])


def test_gini_coefficient_zero_for_uniform_distribution():
    assert gini_coefficient([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0, abs=1e-9)


def test_gini_coefficient_high_for_concentrated_distribution():
    uniform = gini_coefficient([0.25, 0.25, 0.25, 0.25])
    concentrated = gini_coefficient([0.97, 0.01, 0.01, 0.01])
    assert concentrated > uniform


def test_gini_coefficient_zero_for_empty_or_zero_weights():
    assert gini_coefficient([]) == 0.0
    assert gini_coefficient([0.0, 0.0]) == 0.0
