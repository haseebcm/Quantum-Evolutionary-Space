"""Phase 3 -- next-generation Reality Generator: reality operators and families.

`qes.reality_generator.RealityGenerator` already provides two branching
strategies (i.i.d. Gaussian `branch()` and stratified `branch_latin_hypercube()`).
This module adds the richer per-child *operators* the roadmap calls for --
mutation, crossover, interpolation, extrapolation, inversion, structured
perturbation, dimensional transformation, topology transformation,
equation substitution, and parameter transformation -- plus a
`RealityFamily` container that groups a batch of sibling rooms with
family-level statistics, instead of only a flat list of unrelated rooms.

Every operator is a pure function `Room [, Room] -> Room` built on
`Room.clone()` (so lineage, ids, and every untouched field are preserved
correctly) and is deterministic given its RNG/arguments -- there is
nothing "quantum" here, just numpy array transforms.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qes.room import Room

_family_id_counter = itertools.count(1)


def _next_family_id() -> str:
    return f"FAM-{next(_family_id_counter):05d}"


# ---------------------------------------------------------------------------
# Reality operators: each is R_i [, R_j] -> R_ij
# ---------------------------------------------------------------------------


def mutation(room: Room, rng: np.random.Generator, scale: float = 0.05) -> Room:
    """R_ij = G(R_i, zeta): Gaussian perturbation of the state, alias for
    `RealityGenerator.perturb` promoted to a standalone operator."""
    if scale < 0:
        raise ValueError("scale must be >= 0")
    zeta = rng.normal(0.0, scale, size=room.dim)
    return room.clone(x=room.x + zeta)


def crossover(room_a: Room, room_b: Room, rng: np.random.Generator, alpha: float | None = None) -> Room:
    """Blend two parent rooms' states: `alpha * x_a + (1 - alpha) * x_b`.

    A random `alpha` is drawn uniformly in [0, 1] per call if not supplied.
    Both parents must share the same dimensionality.
    """
    if room_a.dim != room_b.dim:
        raise ValueError("crossover requires parents of equal dimensionality")
    a = rng.uniform(0.0, 1.0) if alpha is None else float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    child_x = a * room_a.x + (1.0 - a) * room_b.x
    return room_a.clone(x=child_x, lineage=room_a.lineage + [room_a.id, room_b.id])


def interpolate(room_a: Room, room_b: Room, t: float) -> Room:
    """Linear interpolation between two rooms' states at fraction `t` in [0, 1]."""
    if room_a.dim != room_b.dim:
        raise ValueError("interpolate requires rooms of equal dimensionality")
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1] for interpolation (use extrapolate outside this range)")
    child_x = room_a.x + t * (room_b.x - room_a.x)
    return room_a.clone(x=child_x, lineage=room_a.lineage + [room_a.id, room_b.id])


def extrapolate(room_a: Room, room_b: Room, t: float) -> Room:
    """Linear extrapolation beyond (or before) the `room_a -> room_b` segment.

    `t=0` returns `room_a`'s state, `t=1` returns `room_b`'s state; `t`
    outside [0, 1] projects past either endpoint along the same direction.
    """
    if room_a.dim != room_b.dim:
        raise ValueError("extrapolate requires rooms of equal dimensionality")
    child_x = room_a.x + float(t) * (room_b.x - room_a.x)
    return room_a.clone(x=child_x, lineage=room_a.lineage + [room_a.id, room_b.id])


def inversion(room: Room, center: np.ndarray | None = None) -> Room:
    """Reflect a room's state through `center` (defaults to `room.x_star`)."""
    pivot = room.x_star if center is None else np.asarray(center, dtype=float)
    if pivot.shape != room.x.shape:
        raise ValueError(f"center shape {pivot.shape} != state shape {room.x.shape}")
    return room.clone(x=2.0 * pivot - room.x)


def perturbation(
    room: Room,
    rng: np.random.Generator,
    scale: float = 0.05,
    directions: np.ndarray | None = None,
) -> Room:
    """Structured perturbation along explicit `directions` (unit vectors, one
    per row) rather than isotropic Gaussian noise; falls back to `mutation`
    when `directions` is omitted."""
    if directions is None:
        return mutation(room, rng, scale)
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != room.dim:
        raise ValueError(f"directions must have shape (k, {room.dim})")
    coeffs = rng.normal(0.0, scale, size=directions.shape[0])
    delta = coeffs @ directions
    return room.clone(x=room.x + delta)


def dimensional_transformation(room: Room, matrix: np.ndarray) -> Room:
    """Apply a linear transform `matrix` (shape (dim, dim)) to the room's
    state, target, and bounds -- e.g. a rotation, scaling, or shear of the
    state envelope."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (room.dim, room.dim):
        raise ValueError(f"matrix must have shape ({room.dim}, {room.dim})")
    corners = np.array(
        [matrix @ np.where(mask, room.upper, room.lower) for mask in _corner_masks(room.dim)]
    )
    return room.clone(
        x=matrix @ room.x,
        x_star=matrix @ room.x_star,
        lower=corners.min(axis=0),
        upper=corners.max(axis=0),
        couplings=matrix @ room.couplings @ matrix.T if room.couplings is not None else None,
    )


def _corner_masks(dim: int) -> list[np.ndarray]:
    # 2**dim boolean masks selecting every combination of lower/upper per axis;
    # kept small (only used to re-derive an axis-aligned bounding box after a
    # linear transform) and guarded against combinatorial blow-up.
    if dim > 12:
        raise ValueError("dimensional_transformation only supports dim <= 12 (2**dim corners)")
    return [np.array(bits, dtype=bool) for bits in itertools.product([False, True], repeat=dim)]


def topology_transformation(room: Room, permutation: Sequence[int]) -> Room:
    """Permute the order of the room's state dimensions."""
    perm = np.asarray(permutation, dtype=int)
    if sorted(perm.tolist()) != list(range(room.dim)):
        raise ValueError("permutation must be a permutation of range(room.dim)")
    couplings = room.couplings[np.ix_(perm, perm)] if room.couplings is not None else None
    return room.clone(
        x=room.x[perm],
        x_star=room.x_star[perm],
        lower=room.lower[perm],
        upper=room.upper[perm],
        activation=room.activation[perm],
        couplings=couplings,
    )


def equation_substitution(room: Room, equations: Sequence[Any]) -> Room:
    """Replace a room's active equation population `E_i` wholesale."""
    return room.clone(equations=list(equations))


def parameter_transformation(room: Room, theta_fn: Callable[[dict], dict]) -> Room:
    """Apply `theta_fn` to the room's parameter dict `theta_i`, producing a
    child with transformed (not merely perturbed) model parameters."""
    new_theta = theta_fn(dict(room.theta))
    if not isinstance(new_theta, dict):
        raise TypeError("theta_fn must return a dict")
    return room.clone(theta=new_theta)


# ---------------------------------------------------------------------------
# Reality families: a batch of siblings with family-level statistics
# ---------------------------------------------------------------------------


@dataclass
class RealityFamily:
    """A named batch of sibling rooms produced from one branching operation,
    with aggregate statistics -- instead of only a flat, unrelated room list."""

    members: list[Room]
    parent_id: str | None = None
    id: str = field(default_factory=_next_family_id)

    def __post_init__(self) -> None:
        if not isinstance(self.members, list):
            raise TypeError("members must be a list")

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def mean_state(self) -> np.ndarray:
        """Componentwise mean state across the family."""
        if not self.members:
            raise ValueError("cannot compute mean_state of an empty family")
        return np.mean([m.x for m in self.members], axis=0)

    def std_state(self) -> np.ndarray:
        """Componentwise state standard deviation across the family (spread/diversity)."""
        if not self.members:
            raise ValueError("cannot compute std_state of an empty family")
        return np.std([m.x for m in self.members], axis=0)

    def best(self, score_fn: Callable[[Room], float]) -> Room:
        """Lowest-score (best) member under `score_fn`."""
        if not self.members:
            raise ValueError("cannot select best of an empty family")
        return min(self.members, key=score_fn)

    def diversity(self) -> float:
        """Scalar diversity measure: mean per-dimension standard deviation."""
        if not self.members:
            return 0.0
        return float(np.mean(self.std_state()))


class RealityFamilyBuilder:
    """Convenience wrapper composing the reality operators above into a
    single `RealityFamily`, mixing operator types in one batch (unlike
    `RealityGenerator.branch`, which applies one perturbation style)."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def build(
        self,
        parent: Room,
        count: int,
        mutation_scale: float = 0.05,
        second_parent: Room | None = None,
    ) -> RealityFamily:
        """Generate `count` children mixing mutation, and -- when
        `second_parent` is supplied -- crossover/interpolation/extrapolation
        against it, cycling through operator kinds."""
        operators: list[Callable[[], Room]] = [
            lambda: mutation(parent, self.rng, scale=mutation_scale),
            lambda: inversion(parent),
        ]
        if second_parent is not None:
            operators.extend(
                [
                    lambda: crossover(parent, second_parent, self.rng),
                    lambda: interpolate(parent, second_parent, t=self.rng.uniform(0.0, 1.0)),
                    lambda: extrapolate(parent, second_parent, t=self.rng.uniform(-0.5, 1.5)),
                ]
            )
        members = []
        for i in range(count):
            child = operators[i % len(operators)]()
            child.state = "Active"
            members.append(child)
        return RealityFamily(members=members, parent_id=parent.id)
