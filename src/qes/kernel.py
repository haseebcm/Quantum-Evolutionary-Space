"""Digital Existence Kernel (source-derived; see the PDF-extraction summary,
"QES -- Deep Architecture" / "1. The deepest newly extracted layer: the
Digital Existence Kernel").

An 11-layer closed core, deeper than MCC, ACROS, or COSMIC VP, that builds
existence from the bottom up over a mathematical universe ``S`` of possible
computable states:

    Layer 1  Null Origin           s_0 = empty (nothing instantiated yet)
    Layer 2  Existence Allowance   Delta: empty -> S (definition != instantiation)
    Layer 3  Construction          C: S -> R>=0, bounded accumulation per step
    Layer 4  Mechanism             M: S x S -> {0,1} (identity-preserving transition)
    Layer 5  Operation             O: S -> S with M(s, O(s)) == 1 required
    Layer 6  Rejection             invalid state -> non-existence, not "bad but persistent"
    Layer 7  Persistence           s persists iff lim_{n->inf} O^n(s) == s (materialization)
    Layer 8  Bounded Energy        E(s_t) = (C(s_t+1) - C(s_t)) / dt, 0 < E <= E_max
    Layer 9  Computable Geometry   G(s) = {x : M(s(x), s(x+delta)) == 1}, non-overlapping
    Layer 10 Recursive Entity      L_{t+1} = F(L_t), bounded divergence, reproduction fixed point
    Layer 11 Domain Intelligence   D_{n+1} = Phi(D_n), invariant-preserving domain generation

The key architectural distinction this kernel gives QES: a digital object
does not exist merely because its class or source code exists -- it must
pass through allowance, bounded construction, and identity-preserving
mechanism before being admitted, and an invalid state is rejected into
non-existence rather than generated and deleted later. Other QES layers
(``qes.domain``, ``qes.permission``, ``qes.selection``, ...) can be built on
top of these kernel-level primitives.

This also provides the "reality compiler" distinction from the same
extraction (section 17): given a candidate possibility space ``C``, the
admissible *reality* set is the fixed-point subset

    R = {x in C : x == Phi(x)}

i.e. states that survive their own governing operator unchanged -- see
``RealitySelector``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


def _as_finite_array(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_same_shape(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} shape {a.shape} must match {b_name} shape {b.shape}")


class RejectionError(ValueError):
    """Raised when an operation would produce a non-admissible state.

    Per Layer 6 (Rejection), an inadmissible state is not a "bad but
    persistent object" -- it simply does not come into existence. Callers
    that want the non-raising form should use ``ExistenceKernel.reject``.
    """


@dataclass
class ConstructionResult:
    """Result of one bounded-construction step (Layer 3)."""

    complexity: float
    delta: float


@dataclass
class OperationResult:
    """Result of one governed operation (Layer 5)."""

    state: np.ndarray
    identity_preserved: bool


@dataclass
class PersistenceResult:
    """Result of iterating an operator toward a fixed point (Layer 7)."""

    state: np.ndarray
    converged: bool
    iterations: int
    residual: float


class ExistenceKernel:
    """Layers 1-8 of the Digital Existence Kernel.

    Parameters
    ----------
    epsilon:
        Mechanism tolerance (Layer 4): two states are considered the "same"
        identity iff ``||s_t+1 - s_t|| <= epsilon``.
    delta_max:
        Maximum allowed per-step construction accumulation (Layer 3).
    energy_max:
        Maximum allowed bounded computational energy (Layer 8).
    """

    def __init__(self, epsilon: float = 1e-3, delta_max: float = 1.0, energy_max: float = 1.0):
        epsilon = _as_finite_scalar(epsilon, "epsilon")
        delta_max = _as_finite_scalar(delta_max, "delta_max")
        energy_max = _as_finite_scalar(energy_max, "energy_max")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if delta_max <= 0:
            raise ValueError("delta_max must be > 0")
        if energy_max <= 0:
            raise ValueError("energy_max must be > 0")
        self.epsilon = epsilon
        self.delta_max = delta_max
        self.energy_max = energy_max

    # -- Layer 1: Null Origin -------------------------------------------------
    @staticmethod
    def null_origin(dim: int) -> np.ndarray | None:
        """s_0 = empty. QES represents "no state content exists yet" as
        ``None`` rather than a zero vector -- a zero vector is itself an
        instantiated state, whereas ``None`` is the un-instantiated null
        origin (the ``dim`` argument documents the eventual state's shape
        but is not itself an instantiation)."""
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        if dim < 0:
            raise ValueError("dim must be >= 0")
        return None

    # -- Layer 2: Existence Allowance -----------------------------------------
    @staticmethod
    def allow(candidate: np.ndarray | None) -> np.ndarray:
        """Delta: empty -> S. A digital object does not exist merely
        because its definition exists; it must be explicitly allowed to
        transition from null into an instance."""
        if candidate is None:
            raise RejectionError("cannot allow a null candidate into existence")
        return _as_finite_array(candidate, "candidate")

    # -- Layer 3: Construction -------------------------------------------------
    def construct(self, complexity_t: float, delta: float) -> ConstructionResult:
        """C(s_t+1) = C(s_t) + delta, 0 < delta <= delta_max.

        Enforces *bounded* accumulation rather than unconstrained growth.
        """
        complexity_t = _as_finite_scalar(complexity_t, "complexity_t")
        delta = _as_finite_scalar(delta, "delta")
        if complexity_t < 0:
            raise RejectionError("complexity_t must be >= 0")
        if not (0 < delta <= self.delta_max):
            raise RejectionError(
                f"construction delta {delta} outside bounds (0, {self.delta_max}]"
            )
        return ConstructionResult(complexity=complexity_t + delta, delta=delta)

    # -- Layer 4: Mechanism ----------------------------------------------------
    def is_identity_preserving(self, s_t: np.ndarray, s_t1: np.ndarray) -> bool:
        """M(s_t, s_t+1) = 1 iff ||s_t+1 - s_t|| <= epsilon.

        A world/agent/twin may change while remaining "the same" entity as
        long as the transition stays within this identity tolerance.
        """
        current = _as_finite_array(s_t, "s_t")
        nxt = _as_finite_array(s_t1, "s_t1")
        _validate_same_shape(current, nxt, "s_t", "s_t1")
        return bool(np.linalg.norm(nxt - current) <= self.epsilon)

    # -- Layer 5: Operation ------------------------------------------------------
    def operate(
        self, state: np.ndarray, op_fn: Callable[[np.ndarray], np.ndarray]
    ) -> OperationResult:
        """O: S -> S, valid only when M(s, O(s)) == 1.

        A "valid operation" is a transformation *plus* invariant
        preservation, not a bare function call: if the mechanism check
        fails, the operation is rejected rather than silently applied.
        """
        if not callable(op_fn):
            raise TypeError("op_fn must be callable")
        state_arr = _as_finite_array(state, "state")
        result = _as_finite_array(op_fn(state_arr.copy()), "operation result")
        _validate_same_shape(state_arr, result, "state", "operation result")
        preserved = self.is_identity_preserving(state_arr, result)
        if not preserved:
            raise RejectionError(
                "operation broke identity/coherence "
                f"(||O(s) - s|| > epsilon={self.epsilon})"
            )
        return OperationResult(state=result, identity_preserved=preserved)

    # -- Layer 6: Rejection ------------------------------------------------------
    @staticmethod
    def reject(
        state: np.ndarray, admissible_fn: Callable[[np.ndarray], bool]
    ) -> np.ndarray | None:
        """R(s) = s if s is admissible, else None (non-existence).

        Deeper than deleting a failed simulation: an invalid state simply
        never comes into existence, rather than existing as a "bad but
        persistent object."
        """
        if not callable(admissible_fn):
            raise TypeError("admissible_fn must be callable")
        state_arr = _as_finite_array(state, "state")
        return state_arr if bool(admissible_fn(state_arr)) else None

    # -- Layer 7: Persistence / Materialization ----------------------------------
    def find_persistent_state(
        self,
        state: np.ndarray,
        op_fn: Callable[[np.ndarray], np.ndarray],
        max_iterations: int = 1000,
        tolerance: float = 1e-9,
    ) -> PersistenceResult:
        """s persists iff lim_{n->inf} O^n(s) == s.

        Iterates ``op_fn`` from ``state`` and reports whether it converges
        to a stable fixed point within ``max_iterations`` -- "Persistent
        Entity = Stable Computational Fixed Point," not limited to physical
        materialization.
        """
        if not callable(op_fn):
            raise TypeError("op_fn must be callable")
        _validate_positive_int(max_iterations, "max_iterations")
        tolerance = _as_finite_scalar(tolerance, "tolerance")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")

        current = _as_finite_array(state, "state")
        residual = float("inf")
        for iteration in range(1, max_iterations + 1):
            nxt = _as_finite_array(op_fn(current.copy()), "persistent state result")
            _validate_same_shape(current, nxt, "state", "persistent state result")
            residual = float(np.linalg.norm(nxt - current))
            if residual <= tolerance:
                return PersistenceResult(
                    state=nxt, converged=True, iterations=iteration, residual=residual
                )
            current = nxt
        return PersistenceResult(
            state=current, converged=False, iterations=max_iterations, residual=residual
        )

    # -- Layer 8: Bounded Computational Energy -----------------------------------
    def bounded_energy(self, complexity_t: float, complexity_t1: float, dt: float) -> float:
        """E(s_t) = (C(s_t+1) - C(s_t)) / dt, required in (0, energy_max].

        An internal rate-of-change budget (not physical joules unless a
        domain adapter explicitly maps it), usable to cap how fast any QES
        entity is allowed to construct/change per unit time.
        """
        complexity_t = _as_finite_scalar(complexity_t, "complexity_t")
        complexity_t1 = _as_finite_scalar(complexity_t1, "complexity_t1")
        dt = _as_finite_scalar(dt, "dt")
        if complexity_t < 0 or complexity_t1 < 0:
            raise ValueError("complexity values must be >= 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        energy = (complexity_t1 - complexity_t) / dt
        if not (0 < energy <= self.energy_max):
            raise RejectionError(
                f"computational energy {energy} outside bounds (0, {self.energy_max}]"
            )
        return energy


# -- Layer 9: Computable Geometry ------------------------------------------------
@dataclass
class ComputableGeometry:
    """G(s) = {x : M(s(x), s(x+delta)) == 1}, with non-overlap between
    distinct entities' supports (G(s_i) intersect G(s_j) = empty, i != j).

    Gives QES entities a computable internal geometry (position, topology,
    occupancy, separation) rather than remaining an abstract graph of
    processes.
    """

    support: frozenset[tuple[int, ...]]

    @staticmethod
    def from_grid(
        field_fn: Callable[[tuple[int, ...]], np.ndarray],
        points: Sequence[tuple[int, ...]],
        neighbor_offsets: Sequence[tuple[int, ...]],
        epsilon: float = 1e-3,
    ) -> ComputableGeometry:
        """Build the support set of points ``x`` for which every offset
        neighbor ``x + delta`` is identity-preserving under ``field_fn``."""
        if not callable(field_fn):
            raise TypeError("field_fn must be callable")
        epsilon = _as_finite_scalar(epsilon, "epsilon")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")

        point_list = [tuple(point) for point in points]
        offset_list = [tuple(offset) for offset in neighbor_offsets]
        if not point_list:
            return ComputableGeometry(support=frozenset())

        dimensions = {len(point) for point in point_list}
        if len(dimensions) != 1:
            raise ValueError("points must all have the same dimensionality")
        point_dim = dimensions.pop()
        if any(len(offset) != point_dim for offset in offset_list):
            raise ValueError("neighbor_offsets must match the dimensionality of points")

        cache: dict[tuple[int, ...], np.ndarray] = {}
        exemplar_shape: tuple[int, ...] | None = None

        def evaluate(point: tuple[int, ...]) -> np.ndarray:
            nonlocal exemplar_shape
            if point not in cache:
                value = _as_finite_array(field_fn(point), f"field value at {point!r}")
                if exemplar_shape is None:
                    exemplar_shape = value.shape
                elif value.shape != exemplar_shape:
                    raise ValueError("field_fn must return a consistent array shape for every point")
                cache[point] = value
            return cache[point]

        support: set[tuple[int, ...]] = set()
        for point in point_list:
            fx = evaluate(point)
            if all(
                np.linalg.norm(evaluate(tuple(a + b for a, b in zip(point, offset, strict=True))) - fx)
                <= epsilon
                for offset in offset_list
            ):
                support.add(point)
        return ComputableGeometry(support=frozenset(support))

    def overlaps(self, other: ComputableGeometry) -> bool:
        """True iff this geometry shares any support point with `other`."""
        if not isinstance(other, ComputableGeometry):
            raise TypeError("other must be a ComputableGeometry")
        return bool(self.support & other.support)

    @staticmethod
    def all_disjoint(geometries: Sequence[ComputableGeometry]) -> bool:
        """True iff every pair of geometries has empty intersection."""
        seen: set[tuple[int, ...]] = set()
        for geometry in geometries:
            if not isinstance(geometry, ComputableGeometry):
                raise TypeError("geometries must contain ComputableGeometry instances")
            if seen & geometry.support:
                return False
            seen.update(geometry.support)
        return True


# -- Layer 10: Recursive Autonomous Entities --------------------------------------
@dataclass
class RecursiveEntityResult:
    """Result of stepping a recursive autonomous entity (Layer 10)."""

    state: np.ndarray
    divergence: float
    within_bound: bool


class RecursiveEntity:
    """L_{t+1} = F(L_t), with bounded divergence ||L_t+1 - L_t|| <= lambda,
    and reproduction as a fixed point (exists L' = L such that F(L) = L').

    Gives a rigorous place for autonomous agents, evolving models,
    self-maintaining processes, and world reproduction -- digital-system
    mathematics, not a claim about biological life.
    """

    def __init__(self, lambda_bound: float):
        lambda_bound = _as_finite_scalar(lambda_bound, "lambda_bound")
        if lambda_bound <= 0:
            raise ValueError("lambda_bound must be > 0")
        self.lambda_bound = lambda_bound

    def step(self, state: np.ndarray, f_fn: Callable[[np.ndarray], np.ndarray]) -> RecursiveEntityResult:
        """Advance the recursive entity one step and report the divergence
        from the previous state."""
        if not callable(f_fn):
            raise TypeError("f_fn must be callable")
        state_arr = _as_finite_array(state, "state")
        nxt = _as_finite_array(f_fn(state_arr.copy()), "recursive entity result")
        _validate_same_shape(state_arr, nxt, "state", "recursive entity result")
        div = float(np.linalg.norm(nxt - state_arr))
        return RecursiveEntityResult(state=nxt, divergence=div, within_bound=div <= self.lambda_bound)

    def reproduces(
        self,
        state: np.ndarray,
        f_fn: Callable[[np.ndarray], np.ndarray],
        tolerance: float = 1e-9,
    ) -> bool:
        """True iff ``state`` is (approximately) a fixed point of ``f_fn``,
        i.e. reproduction: exists L' = L such that F(L) = L'."""
        if not callable(f_fn):
            raise TypeError("f_fn must be callable")
        tolerance = _as_finite_scalar(tolerance, "tolerance")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        state_arr = _as_finite_array(state, "state")
        nxt = _as_finite_array(f_fn(state_arr.copy()), "recursive reproduction result")
        _validate_same_shape(state_arr, nxt, "state", "recursive reproduction result")
        return bool(np.linalg.norm(nxt - state_arr) <= tolerance)


# -- Layer 11: Self-Generating Domain Intelligence --------------------------------
@dataclass
class DomainGenerationResult:
    """Result of generating one new domain generation (Layer 11)."""

    domains: tuple[np.ndarray, ...]
    all_admissible: bool


class SelfGeneratingDomainIntelligence:
    """D_{n+1} = Phi(D_n), subject to invariant preservation: for all d in
    D_n, Phi(d) in A (the admissible set).

    QES can generate new computational domains without changing its core
    admissibility laws -- the universe expands while the kernel stays
    invariant (5 -> 33 -> 55+ -> 338+ -> 8B+ scaling is realized through
    iterated constrained divergence like this, not by adding arbitrary
    domains).
    """

    def __init__(self, admissible_fn: Callable[[np.ndarray], bool]):
        if not callable(admissible_fn):
            raise TypeError("admissible_fn must be callable")
        self.admissible_fn = admissible_fn

    def generate(
        self, domains: Sequence[np.ndarray], phi_fn: Callable[[np.ndarray], np.ndarray]
    ) -> DomainGenerationResult:
        """Generate the next domain population and report whether every
        generated domain remains admissible."""
        if not callable(phi_fn):
            raise TypeError("phi_fn must be callable")

        next_domains: list[np.ndarray] = []
        all_admissible = True
        for domain in domains:
            current = _as_finite_array(domain, "domain")
            nxt = _as_finite_array(phi_fn(current.copy()), "generated domain")
            _validate_same_shape(current, nxt, "domain", "generated domain")
            next_domains.append(nxt)
            all_admissible = all_admissible and bool(self.admissible_fn(nxt))
        return DomainGenerationResult(domains=tuple(next_domains), all_admissible=all_admissible)


# -- Reality compiler: candidates vs realities (section 17) -----------------------
@dataclass
class RealitySelectionResult:
    """Result of filtering a candidate set down to admissible realities."""

    realities: list[np.ndarray] = field(default_factory=list)
    rejected: list[np.ndarray] = field(default_factory=list)


class RealitySelector:
    """Distinguishes candidate possibilities ``C`` from instantiated
    realities ``R = {x in C : x == Phi(x)}`` -- states that are fixed
    points of the governing operator survive; all others remain
    candidates only (never instantiated).

    This realizes the "fixed-point merge" architecture: generation,
    filtering, and realization are not independent stages -- stable
    existence is itself the result of satisfying the complete operator.
    """

    def __init__(self, phi_fn: Callable[[np.ndarray], np.ndarray], tolerance: float = 1e-6):
        if not callable(phi_fn):
            raise TypeError("phi_fn must be callable")
        tolerance = _as_finite_scalar(tolerance, "tolerance")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        self.phi_fn = phi_fn
        self.tolerance = tolerance

    def is_reality(self, candidate: np.ndarray) -> bool:
        """x == Phi(x) within tolerance."""
        candidate_arr = _as_finite_array(candidate, "candidate")
        next_state = _as_finite_array(self.phi_fn(candidate_arr.copy()), "phi result")
        _validate_same_shape(candidate_arr, next_state, "candidate", "phi result")
        return bool(np.linalg.norm(next_state - candidate_arr) <= self.tolerance)

    def select(self, candidates: Sequence[np.ndarray]) -> RealitySelectionResult:
        """Partition `candidates` into fixed-point realities and rejected
        non-realities."""
        result = RealitySelectionResult()
        for candidate in candidates:
            candidate_arr = _as_finite_array(candidate, "candidate")
            target = result.realities if self.is_reality(candidate_arr) else result.rejected
            target.append(candidate_arr)
        return result
