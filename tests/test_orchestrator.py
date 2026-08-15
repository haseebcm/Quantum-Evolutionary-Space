import numpy as np
import pytest

from qes.orchestrator import Acros, AdaptiveAcros, RoomLifecycle
from qes.room import Room


def make_room(n=2, state="Seed"):
    room = Room(
        x=np.zeros(n),
        x_star=np.zeros(n),
        lower=-np.ones(n),
        upper=np.ones(n),
        activation=np.ones(n),
    )
    room.state = state
    return room


def test_can_transition_seed_to_active():
    assert RoomLifecycle.can_transition("Seed", "Active")


def test_cannot_transition_collapsed_to_active():
    assert not RoomLifecycle.can_transition("Collapsed", "Active")


def test_can_transition_raises_on_unknown_state():
    with pytest.raises(ValueError):
        RoomLifecycle.can_transition("Seed", "Bogus")


def test_transition_updates_room_state():
    room = make_room(state="Seed")
    RoomLifecycle.transition(room, "Active")
    assert room.state == "Active"


def test_transition_raises_on_illegal_move():
    room = make_room(state="Collapsed")
    with pytest.raises(ValueError):
        RoomLifecycle.transition(room, "Active")


def test_merge_averages_state_and_combines_lineage():
    a = make_room(state="Active")
    a.x = np.array([0.0, 0.0])
    b = make_room(state="Active")
    b.x = np.array([2.0, 4.0])
    merged = RoomLifecycle.merge(a, b)
    np.testing.assert_allclose(merged.x, [1.0, 2.0])
    assert a.id in merged.lineage
    assert b.id in merged.lineage
    assert merged.state == "Merged"


def test_acros_correction_scaled_by_chi_and_gain():
    def grad_fn(x, ctx):
        return np.array([1.0, 1.0])

    acros = Acros(gradient_fn=grad_fn, gain=2.0)
    correction_on = acros.correction(np.zeros(2), chi=1.0)
    correction_off = acros.correction(np.zeros(2), chi=0.0)
    np.testing.assert_allclose(correction_on, [2.0, 2.0])
    np.testing.assert_allclose(correction_off, [0.0, 0.0])


def test_acros_state_derivative_combines_drift_and_correction():
    def grad_fn(x, ctx):
        return np.array([1.0])

    def drift_fn(x, t):
        return np.array([5.0])

    acros = Acros(gradient_fn=grad_fn, gain=1.0)
    xdot = acros.state_derivative(np.zeros(1), t=0.0, drift_fn=drift_fn, chi=1.0)
    np.testing.assert_allclose(xdot, [4.0])


def test_meta_adapt_selects_candidate_minimizing_combined_cost():
    candidates = ["a", "b", "c"]
    costs = {"a": (1.0, 1.0, 1.0), "b": (0.1, 0.1, 0.1), "c": (2.0, 2.0, 2.0)}
    best = Acros.meta_adapt(
        candidates,
        risk_fn=lambda c: costs[c][0],
        inconsistency_fn=lambda c: costs[c][1],
        instability_fn=lambda c: costs[c][2],
    )
    assert best == "b"


def test_adaptive_acros_correction_uses_bias_corrected_moments_on_first_step():
    def grad_fn(x, ctx):
        return np.array([1.0, 1.0])

    acros = AdaptiveAcros(gradient_fn=grad_fn, gain=1.0)
    correction = acros.correction(np.zeros(2), chi=1.0)
    # After bias correction, m_hat == v_hat == grad on the very first step,
    # so correction == gain * grad / (sqrt(grad^2) + eps) ~= gain * sign(grad).
    np.testing.assert_allclose(correction, [1.0, 1.0], atol=1e-4)


def test_adaptive_acros_accumulates_moments_across_calls():
    def grad_fn(x, ctx):
        return np.array([2.0])

    acros = AdaptiveAcros(gradient_fn=grad_fn, gain=1.0)
    acros.correction(np.zeros(1), chi=1.0)
    acros.correction(np.zeros(1), chi=1.0)
    assert acros._t == 2
    assert acros._m is not None


def test_adaptive_acros_reset_clears_moment_state():
    def grad_fn(x, ctx):
        return np.array([2.0])

    acros = AdaptiveAcros(gradient_fn=grad_fn)
    acros.correction(np.zeros(1), chi=1.0)
    acros.reset()
    assert acros._m is None
    assert acros._v is None
    assert acros._t == 0
