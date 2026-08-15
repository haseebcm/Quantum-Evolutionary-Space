import numpy as np
import pytest

from qes.domain import DomainNullification


def test_metric_combines_weighted_domain_matrices():
    w1 = np.eye(2)
    w2 = 2 * np.eye(2)
    dn = DomainNullification([w1, w2])
    combined = dn.metric(np.array([1.0, 0.5]))
    np.testing.assert_allclose(combined, np.eye(2) * (1.0 + 1.0))


def test_metric_all_zero_activation_gives_zero_matrix():
    dn = DomainNullification([np.eye(2), np.eye(2)])
    combined = dn.metric(np.array([0.0, 0.0]))
    np.testing.assert_allclose(combined, np.zeros((2, 2)))


def test_effective_state_applies_activation_elementwise():
    x = np.array([2.0, 4.0, 6.0])
    a = np.array([1.0, 0.0, 0.5])
    np.testing.assert_allclose(
        DomainNullification.effective_state(a, x), [2.0, 0.0, 3.0]
    )


def test_activation_matrix_is_diagonal():
    a = np.array([1.0, 2.0, 3.0])
    mat = DomainNullification.activation_matrix(a)
    np.testing.assert_allclose(mat, np.diag(a))


def test_compose_concatenates_domain_metrics_from_both_sources():
    dn_a = DomainNullification([np.eye(2)])
    dn_b = DomainNullification([2 * np.eye(2), 3 * np.eye(2)])
    composed = dn_a.compose(dn_b)
    assert len(composed.domain_metrics) == 3
    combined = composed.metric(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(combined, np.eye(2) * (1.0 + 2.0 + 3.0))


def test_compose_does_not_mutate_original_instances():
    dn_a = DomainNullification([np.eye(2)])
    dn_b = DomainNullification([2 * np.eye(2)])
    dn_a.compose(dn_b)
    assert len(dn_a.domain_metrics) == 1
    assert len(dn_b.domain_metrics) == 1


def test_constructor_rejects_more_than_33_domains():
    with pytest.raises(ValueError):
        DomainNullification([np.eye(2)] * 34)


def test_metric_rejects_mismatched_activation_length():
    dn = DomainNullification([np.eye(2), np.eye(2)])
    with pytest.raises(ValueError):
        dn.metric(np.array([1.0, 1.0, 1.0]))
