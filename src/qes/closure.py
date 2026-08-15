"""Formal closure and cross-domain verification (docs/QES-architecture.md
PDF-extraction summary, sections O-P).

A safety gate says "do not permit this." A closure layer says: every
subsequent transformation must remain representable, admissible, and
internally governed. This module implements the source's closure axioms:

    Divergence closure:   D(x) = inf_{v in V} ||x - v||,  D(x) = 0 <=> x in V
    Collapse proximity:   S(x) = phi(D(x), ||grad D(x)||, C(x))
    Permission closure:   A(x) = { u in A : G(x, u) <= 0 }
    Safe-control:         for all x in X_op, A(x) != empty
    Action selection:     pi(x) in argmin_{u in A(x)} [ J(x,u) + lambda*S(F(x,u,xi)) ]
    Safe exploration:     the policy-search operator maps admissible
                          policies back to admissible policies

and the cross-domain verification interface (section P):

    Encoder:              E_k : Y_k -> X_k
    Reconstruction:        y_k ~= y_hat_k(E_k(y_k))
    State closure:         F_k(x, u, xi) in X
    Constraint closure:    u in A(x) <=> G(x, u) <= 0
    Risk monotonicity:     S(x) increases as known failure boundaries are approached
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_finite_vector(value: object, *, name: str) -> np.ndarray:
    """Return `value` as a finite 1-D float array."""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a one-dimensional float array") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_finite_scalar(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return `value` as a finite float, optionally range-checked."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real-valued scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and scalar < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return scalar


def _stack_viable_set(viable_set: Sequence[np.ndarray], reference: np.ndarray) -> np.ndarray:
    """Validate and stack a sampled viable set into shape `(n_points, dim)`."""
    if not viable_set:
        raise ValueError("viable_set must be non-empty")
    vectors: list[np.ndarray] = []
    for index, point in enumerate(viable_set):
        vector = _as_finite_vector(point, name=f"viable_set[{index}]")
        if vector.shape != reference.shape:
            raise ValueError(
                f"viable_set[{index}] shape {vector.shape} does not match x shape {reference.shape}"
            )
        vectors.append(vector)
    return np.vstack(vectors)


def _minimum_distances(query_points: np.ndarray, viable_points: np.ndarray) -> np.ndarray:
    """Return the minimum Euclidean distance from each query point to `viable_points`."""
    deltas = query_points[:, np.newaxis, :] - viable_points[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return np.min(distances, axis=1)


def divergence(x: np.ndarray, viable_set: Sequence[np.ndarray]) -> float:
    """D(x) = inf_{v in V} ||x - v||, approximated over a finite/sampled
    representation of the viable set V (the true set may be infinite or
    only implicitly known; a finite sample is the practical realization).

    D(x) == 0 iff x coincides with a point of `viable_set` (x in V)."""
    x_vector = _as_finite_vector(x, name="x")
    viable_points = _stack_viable_set(viable_set, x_vector)
    return float(_minimum_distances(x_vector[np.newaxis, :], viable_points)[0])


def divergence_gradient(
    x: np.ndarray, viable_set: Sequence[np.ndarray], eps: float = 1e-6
) -> np.ndarray:
    """Numerical gradient of `divergence` at `x` (central differences),
    used by `collapse_proximity` for ||grad D(x)||."""
    x_vector = _as_finite_vector(x, name="x")
    step = _as_finite_scalar(eps, name="eps", minimum=0.0)
    if step == 0.0:
        raise ValueError("eps must be > 0")
    if x_vector.size == 0:
        return np.zeros_like(x_vector)

    viable_points = _stack_viable_set(viable_set, x_vector)
    basis = np.eye(x_vector.size, dtype=float) * step
    plus_points = x_vector + basis
    minus_points = x_vector - basis
    plus_distances = _minimum_distances(plus_points, viable_points)
    minus_distances = _minimum_distances(minus_points, viable_points)
    return (plus_distances - minus_distances) / (2.0 * step)


def collapse_proximity(d: float, grad_norm: float, c: float) -> float:
    """S(x) = phi(D(x), ||grad D(x)||, C(x)): a weighted combination of
    distance-to-viable, its local rate of change, and a caller-supplied
    context/criticality term `c` -- the collapse-proximity score used by
    the action-selection closure below."""
    distance = _as_finite_scalar(d, name="d", minimum=0.0)
    gradient_norm = _as_finite_scalar(grad_norm, name="grad_norm", minimum=0.0)
    criticality = _as_finite_scalar(c, name="c")
    return distance + gradient_norm + criticality


@dataclass
class PermissionClosureResult:
    """Output of `PermissionClosure.evaluate()`."""

    admissible_actions: list[object]
    is_safe: bool  # non-empty safe-control condition


class PermissionClosure:
    """A(x) = { u in A : G(x, u) <= 0 }, and the non-empty safe-control
    condition: for all x in the operating region, A(x) must be non-empty."""

    def __init__(
        self,
        candidate_actions: Sequence[object],
        constraint_fn: Callable[[object, object], float],
    ) -> None:
        """Store the candidate action set and the scalar constraint function."""
        if not callable(constraint_fn):
            raise TypeError("constraint_fn must be callable")
        self.candidate_actions = list(candidate_actions)
        self.constraint_fn = constraint_fn

    def evaluate(self, x: object) -> PermissionClosureResult:
        """Filter candidate actions down to the admissible set A(x)."""
        admissible: list[object] = []
        for action in self.candidate_actions:
            constraint_value = _as_finite_scalar(
                self.constraint_fn(x, action),
                name=f"constraint_fn(x, {action!r})",
            )
            if constraint_value <= 0.0:
                admissible.append(action)
        return PermissionClosureResult(admissible_actions=admissible, is_safe=bool(admissible))


def action_selection(
    x: object,
    actions: Sequence[object],
    j_fn: Callable[[object, object], float],
    s_fn: Callable[[object], float],
    transition_fn: Callable[[object, object], object],
    lam: float = 1.0,
) -> object:
    """pi(x) in argmin_{u in A(x)} [ J(x,u) + lambda * S(F(x,u,xi)) ].

    `actions` is treated as the already-admissible set A(x) (callers should
    filter via `PermissionClosure` first); `transition_fn(x, u)` realizes
    F(x, u, xi) for the (possibly stochastic, caller-controlled) xi.
    """
    if not callable(j_fn):
        raise TypeError("j_fn must be callable")
    if not callable(s_fn):
        raise TypeError("s_fn must be callable")
    if not callable(transition_fn):
        raise TypeError("transition_fn must be callable")
    if not actions:
        raise ValueError("actions (A(x)) must be non-empty")
    lam_value = _as_finite_scalar(lam, name="lam", minimum=0.0)

    best_action: object | None = None
    best_score: float | None = None
    for action in actions:
        next_state = transition_fn(x, action)
        stage_cost = _as_finite_scalar(j_fn(x, action), name="j_fn(x, action)")
        risk_cost = _as_finite_scalar(
            s_fn(next_state),
            name="s_fn(transition_fn(x, action))",
        )
        total_score = stage_cost + lam_value * risk_cost
        if best_score is None or total_score < best_score:
            best_action = action
            best_score = total_score
    if best_action is None:
        raise ValueError("actions (A(x)) must be non-empty")
    return best_action


def safe_exploration_closure(
    policies: Sequence[object],
    search_operator: Callable[[object], object],
    is_admissible_fn: Callable[[object], bool],
) -> bool:
    """True iff `search_operator` maps every admissible policy in
    `policies` back to an admissible policy (the safe exploration closure
    property: the policy-search step never leaves the admissible set)."""
    if not callable(search_operator):
        raise TypeError("search_operator must be callable")
    if not callable(is_admissible_fn):
        raise TypeError("is_admissible_fn must be callable")
    for policy in policies:
        if bool(is_admissible_fn(policy)) and not bool(is_admissible_fn(search_operator(policy))):
            return False
    return True


def _default_error_fn(a: object, b: object) -> float:
    a_vector = _as_finite_vector(a, name="a")
    b_vector = _as_finite_vector(b, name="b")
    if a_vector.shape != b_vector.shape:
        raise ValueError(f"a shape {a_vector.shape} does not match b shape {b_vector.shape}")
    return float(np.linalg.norm(a_vector - b_vector))


@dataclass
class DomainVerificationResult:
    """Result of one `CrossDomainVerifier.verify()` call for one domain set."""

    reconstruction_error: float
    state_closed: bool
    constraint_closed: bool


class CrossDomainVerifier:
    """The QES World-to-State Verification Interface (section P): for each
    domain set k, an encoder E_k maps measured reality Y_k into the QES
    state space X_k, and the source expects three closures: reconstruction
    consistency, state closure, and constraint closure, plus risk
    monotonicity as failure boundaries are approached."""

    def __init__(
        self,
        encoder: Callable[[object], np.ndarray],
        decoder: Callable[[np.ndarray], object],
        transition_fn: Callable[[np.ndarray, object, object], np.ndarray],
        state_space_check: Callable[[np.ndarray], bool],
        constraint_fn: Callable[[np.ndarray, object], float],
    ) -> None:
        """Store the domain-specific encode/decode/closure hooks."""
        for name, fn in (
            ("encoder", encoder),
            ("decoder", decoder),
            ("transition_fn", transition_fn),
            ("state_space_check", state_space_check),
            ("constraint_fn", constraint_fn),
        ):
            if not callable(fn):
                raise TypeError(f"{name} must be callable")
        self.encoder = encoder
        self.decoder = decoder
        self.transition_fn = transition_fn
        self.state_space_check = state_space_check
        self.constraint_fn = constraint_fn

    def verify(
        self,
        y: object,
        u: object,
        xi: object,
        error_fn: Callable[[object, object], float] = _default_error_fn,
    ) -> DomainVerificationResult:
        """Encode `y`, decode back (reconstruction check), advance one
        transition (state closure check), and check the constraint gate
        (constraint closure: u in A(x) iff G(x, u) <= 0) at the encoded state."""
        if not callable(error_fn):
            raise TypeError("error_fn must be callable")
        encoded = _as_finite_vector(self.encoder(y), name="encoder(y)")
        reconstructed = self.decoder(encoded)
        reconstruction_error = _as_finite_scalar(
            error_fn(y, reconstructed),
            name="error_fn(y, reconstructed)",
        )

        next_state = _as_finite_vector(
            self.transition_fn(encoded, u, xi),
            name="transition_fn(encoded, u, xi)",
        )
        if next_state.shape != encoded.shape:
            raise ValueError(
                f"transition_fn(encoded, u, xi) shape {next_state.shape} does not match "
                f"encoded shape {encoded.shape}"
            )
        state_closed = bool(self.state_space_check(next_state))

        constraint_closed = _as_finite_scalar(
            self.constraint_fn(encoded, u),
            name="constraint_fn(encoded, u)",
        ) <= 0.0

        return DomainVerificationResult(
            reconstruction_error=reconstruction_error,
            state_closed=state_closed,
            constraint_closed=constraint_closed,
        )

    @staticmethod
    def risk_monotonicity(proximities: Sequence[float]) -> bool:
        """True iff S(x) is non-decreasing across a sequence of states that
        approach a known failure boundary (risk must not spuriously drop)."""
        values = np.asarray(list(proximities), dtype=float)
        if values.ndim != 1:
            raise ValueError("proximities must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError("proximities must contain only finite values")
        if values.size < 2:
            return True
        return bool(np.all(np.diff(values) >= 0.0))
