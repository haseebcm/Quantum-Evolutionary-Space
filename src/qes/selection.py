"""The Genesis–Selection theorem and solution kinds (docs/QES-architecture.md, sections 18-19).

Layered search:
    A_k     = intersection_{j=1..k} { x in U : G_j(x) <= 0 }
    P_k     = Gamma_k(A_k)                                   (generate candidates)
    K_k(s)  = { p in P_k : sigma_k(p) = s }                  (classify by signature)
    J_k(p)  = alpha_k*D_k(p) + beta_k*S_k(p) - gamma_k*V_k(p) (score, lower is better)
    p_k*(s) = argmin_{p in K_k(s)} J_k(p)                    (select)

Solution kinds (section 19): group candidates by signature and keep the best
survivor per kind, preserving fundamentally different solution families
instead of collapsing everything into one answer.

Multi-objective extension: `pareto_front()` generalizes the single-score
argmin to a non-dominated set under several simultaneous objectives, and
`GenesisSelection.select_pareto_per_kind()` combines this with signature
grouping to keep an entire Pareto front per solution kind rather than one
scalar-best survivor.

Full 11-layer pipeline (PDF-extraction summary, section D, "The
Genesis-Selection pipeline is its own 11-layer governance architecture"):
`GenesisSelectionPipeline` runs the complete per-layer stage sequence --
generate candidates X_k, extract signatures, detect first-of-kind
products (`sigma_1(p) not in Sigma_known,1`), apply the admissibility
filter, select a supreme survivor per surviving kind, and advance to the
next (strictly tightening) layer -- tracking the population-shrinking
invariant `P_11 subseteq ... subseteq P_1` and terminating a kind either
when it reaches a final Layer-11 representative or when it is eliminated
at the first layer it cannot satisfy.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field

SignatureFn = Callable[[object], Hashable]
GateFn = Callable[[object], float]
ScoreFn = Callable[[object], float]
ObjectivesFn = Callable[[object], Sequence[float]]
GeneratorFn = Callable[[Sequence[object]], Sequence[object]]


@dataclass
class ScoredCandidate:
    candidate: object
    signature: Hashable
    score: float


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True iff objective vector `a` Pareto-dominates `b` (all objectives to
    be minimized): `a` is no worse in every objective and strictly better in
    at least one."""
    not_worse = all(x <= y for x, y in zip(a, b, strict=True))
    strictly_better = any(x < y for x, y in zip(a, b, strict=True))
    return not_worse and strictly_better


def pareto_front(candidates: Sequence[object], objectives_fn: ObjectivesFn) -> list:
    """Return the non-dominated subset of `candidates` under `objectives_fn`
    (each candidate mapped to a tuple of objectives to be minimized).

    This generalizes the single-score `J_k` argmin selection (section 18) to
    multi-objective search: instead of collapsing every criterion into one
    scalar score, keep every candidate that is not strictly beaten on all
    fronts simultaneously by another candidate.
    """
    scored = [(c, tuple(objectives_fn(c))) for c in candidates]
    front = []
    for candidate, objectives in scored:
        if not any(
            other_obj != objectives and dominates(other_obj, objectives)
            for _, other_obj in scored
        ):
            front.append(candidate)
    return front


class GenesisSelection:
    """Applies successive constraint gates, classifies by signature, and selects survivors."""

    def __init__(
        self,
        gates: Sequence[GateFn],
        signature_fn: SignatureFn,
        score_fn: ScoreFn,
    ):
        """
        Args:
            gates: sequence of constraint functions G_j(x) -> float; a
                candidate passes layer k if G_j(x) <= 0 for all j <= k.
            signature_fn: sigma(p) -> hashable signature used to group
                candidates into solution kinds K(s).
            score_fn: J(p) -> float; lower score is preferred by argmin selection.
        """
        self.gates = list(gates)
        self.signature_fn = signature_fn
        self.score_fn = score_fn

    def admissible(self, candidate: object, layer: int) -> bool:
        """True iff candidate satisfies all gates up to and including `layer` (0-indexed)."""
        return all(self.gates[j](candidate) <= 0 for j in range(layer + 1))

    def filter_layer(self, candidates: Sequence[object], layer: int) -> list:
        """A_k restricted to the supplied candidate generator's output P_k."""
        return [c for c in candidates if self.admissible(c, layer)]

    def classify(self, candidates: Sequence[object]) -> dict:
        """K(s) = { R_i : sigma(R_i) = s } for all candidates."""
        classes: dict = {}
        for c in candidates:
            s = self.signature_fn(c)
            classes.setdefault(s, []).append(c)
        return classes

    def select_supreme(self, candidates: Sequence[object]) -> dict:
        """Return the best (lowest-score) survivor per signature/kind.

        Returns:
            dict mapping signature -> ScoredCandidate for the argmin survivor
            of that kind.
        """
        classes = self.classify(candidates)
        survivors = {}
        for signature, members in classes.items():
            best = min(members, key=self.score_fn)
            survivors[signature] = ScoredCandidate(
                candidate=best, signature=signature, score=self.score_fn(best)
            )
        return survivors

    def run(self, candidates: Sequence[object]) -> dict:
        """Apply all gate layers in sequence, then select supreme survivors per kind."""
        surviving = list(candidates)
        for layer in range(len(self.gates)):
            surviving = self.filter_layer(surviving, layer)
            if not surviving:
                break
        return self.select_supreme(surviving)

    def select_pareto_per_kind(
        self, candidates: Sequence[object], objectives_fn: ObjectivesFn
    ) -> dict:
        """Return the Pareto-non-dominated survivors per signature/kind.

        Unlike `select_supreme` (single-score argmin), this keeps every
        non-dominated candidate within each solution kind under
        `objectives_fn`'s multi-objective vector, preserving trade-off
        fronts instead of collapsing to one scalar-best survivor.
        """
        classes = self.classify(candidates)
        return {
            signature: pareto_front(members, objectives_fn)
            for signature, members in classes.items()
        }


@dataclass
class KindTrace:
    """Per-kind trace across the 11-layer Genesis-Selection pipeline."""

    signature: Hashable
    first_seen_layer: int
    survivor: object | None = None
    score: float | None = None
    eliminated_at_layer: int | None = None
    population_sizes: list = field(default_factory=list)


@dataclass
class PipelineResult:
    """Final outcome of `GenesisSelectionPipeline.run()`."""

    kinds: dict  # signature -> KindTrace
    population_history: list  # population size after each layer (monotone non-increasing)


class GenesisSelectionPipeline:
    """The full 11-stage Genesis-Selection governance pipeline.

    Unlike `GenesisSelection.run()` (a single generate-then-filter pass),
    this drives the complete per-layer sequence described in the source:
    generate candidates for the layer, extract signatures, detect
    first-of-kind products, apply the admissibility filter (any caller
    gate, e.g. a `GenesisPermission`/HSA check), select a supreme survivor
    per surviving kind, and advance -- enforcing that candidate
    populations can only shrink layer over layer
    (`P_11 subseteq ... subseteq P_1`).
    """

    def __init__(
        self,
        gates: Sequence[GateFn],
        signature_fn: SignatureFn,
        score_fn: ScoreFn,
    ):
        """
        Args:
            gates: one admissibility gate per layer k=1..len(gates); a
                candidate survives layer k iff `gates[k](candidate) <= 0`.
            signature_fn: sigma_k(p) -> hashable signature/kind.
            score_fn: J_k(p) -> float, lower is better (used by the
                supreme-selection argmin within each surviving kind).
        """
        if not gates:
            raise ValueError("gates must be non-empty (at least one layer)")
        self.gates = list(gates)
        self.signature_fn = signature_fn
        self.score_fn = score_fn

    def run(
        self,
        initial_candidates: Sequence[object],
        generators: Sequence[GeneratorFn] | None = None,
    ) -> PipelineResult:
        """Run all layers in sequence.

        Args:
            initial_candidates: the seed candidate population X_1.
            generators: optional per-layer candidate generator
                `Gamma_k(surviving) -> new_candidates`, applied *before*
                that layer's gate (e.g. mutating/expanding survivors into
                new products for that layer). If omitted, the previous
                layer's survivors are reused unchanged as that layer's
                input population (a pure filter-and-select pipeline).
        """
        population = list(initial_candidates)
        known_signatures_layer1: set = set()
        kinds: dict[Hashable, KindTrace] = {}
        population_history: list[int] = []

        for layer_index, gate in enumerate(self.gates, start=1):
            if generators is not None and layer_index - 1 < len(generators):
                population = list(generators[layer_index - 1](population))

            for candidate in population:
                signature = self.signature_fn(candidate)
                if signature not in kinds:
                    kinds[signature] = KindTrace(signature=signature, first_seen_layer=layer_index)
                    if layer_index == 1:
                        known_signatures_layer1.add(signature)

            surviving = [c for c in population if gate(c) <= 0]
            alive_signatures = {self.signature_fn(c) for c in surviving}
            for trace in list(kinds.values()):
                if trace.eliminated_at_layer is None and trace.survivor is None:
                    if trace.signature not in alive_signatures and trace.first_seen_layer <= layer_index:
                        trace.eliminated_at_layer = layer_index

            for trace in kinds.values():
                trace.population_sizes.append(len(surviving))

            population = surviving
            population_history.append(len(population))
            if not population:
                break

        if population:
            final_classes: dict[Hashable, list] = {}
            for candidate in population:
                final_classes.setdefault(self.signature_fn(candidate), []).append(candidate)

            for signature, members in final_classes.items():
                best = min(members, key=self.score_fn)
                trace = kinds[signature]
                trace.survivor = best
                trace.score = self.score_fn(best)

        if generators is None:
            for earlier, later in zip(population_history, population_history[1:], strict=False):
                if later > earlier:  # pragma: no cover - impossible without generators because each layer only filters survivors
                    raise RuntimeError("candidate population must not grow without a generator")

        return PipelineResult(kinds=kinds, population_history=population_history)
