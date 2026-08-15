"""Phase 13 -- heuristic meta-learning for QES search strategy selection.

This module makes QES adapt *how* it searches, not just how many children one
generation spawns. The "meta-learning" here is intentionally modest and
explicit: it is a documented heuristic over cheap problem features plus tracked
past outcomes, not literal deep learning, AGI, or free compute.

`MetaController.recommend()` starts from sensible defaults, adjusts them from a
problem-structure feature vector, then nudges the recommendation toward
historically successful configurations observed on similar problems. Branching
factor reuse is delegated to `qes.cycle.adaptive_branch_count()` so the same
uncertainty/risk/novelty/compute-budget reasoning remains consistent with the
Phase 2 autonomous cycle.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field

import numpy as np

from qes.cycle import adaptive_branch_count
from qes.patterns import Pattern, PatternMemory

_outcome_id_counter = itertools.count(1)


def _next_outcome_id() -> str:
    return f"meta-outcome-{next(_outcome_id_counter):05d}"


def _validate_unit_interval(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return scalar


def _validate_positive_float(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return scalar


def _validate_non_negative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _expect_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _mapping_int(data: Mapping[str, object], key: str, default: int | None = None) -> int:
    raw = data[key] if key in data else default
    if raw is None:
        raise KeyError(key)
    if isinstance(raw, bool):
        raise TypeError(f"{key} must be an integer, not a boolean")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{key} must be int-compatible")
    return int(raw)


def _mapping_float(data: Mapping[str, object], key: str, default: float | None = None) -> float:
    raw = data[key] if key in data else default
    if raw is None:
        raise KeyError(key)
    if isinstance(raw, bool):
        raise TypeError(f"{key} must be float-compatible, not a boolean")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{key} must be float-compatible")
    return float(raw)


@dataclass
class ProblemStructure:
    """Cheap heuristic features describing a search problem.

    These fields are not a rigorous problem taxonomy; they are an honest,
    low-cost feature vector that callers can estimate before a run starts so
    QES can choose a more appropriate search strategy.
    """

    dimensionality: int
    bounds_width: float
    ruggedness: float = 0.5
    noise_level: float = 0.0
    constraint_count: int = 0
    gradient_available: bool = False

    def __post_init__(self) -> None:
        self.dimensionality = _validate_positive_int("dimensionality", self.dimensionality)
        self.bounds_width = _validate_positive_float("bounds_width", self.bounds_width)
        self.ruggedness = _validate_unit_interval("ruggedness", self.ruggedness)
        self.noise_level = _validate_unit_interval("noise_level", self.noise_level)
        self.constraint_count = _validate_non_negative_int("constraint_count", self.constraint_count)
        if not isinstance(self.gradient_available, bool):
            raise TypeError("gradient_available must be a boolean")

    def feature_vector(self) -> np.ndarray:
        """Return a normalized feature vector for nearest-neighbor comparisons."""
        dim_signal = _clamp(math.log1p(self.dimensionality) / math.log1p(128.0), 0.0, 1.0)
        width_signal = _clamp(self.bounds_width / (self.bounds_width + 10.0), 0.0, 1.0)
        constraint_signal = _clamp(self.constraint_count / (self.constraint_count + 8.0), 0.0, 1.0)
        return np.array(
            [
                dim_signal,
                width_signal,
                self.ruggedness,
                self.noise_level,
                constraint_signal,
                1.0 if self.gradient_available else 0.0,
            ],
            dtype=float,
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize this feature vector to a plain mapping."""
        return {
            "dimensionality": self.dimensionality,
            "bounds_width": self.bounds_width,
            "ruggedness": self.ruggedness,
            "noise_level": self.noise_level,
            "constraint_count": self.constraint_count,
            "gradient_available": self.gradient_available,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> ProblemStructure:
        """Build a `ProblemStructure` from a plain mapping."""
        return cls(
            dimensionality=_mapping_int(data, "dimensionality"),
            bounds_width=_mapping_float(data, "bounds_width"),
            ruggedness=_mapping_float(data, "ruggedness", 0.5),
            noise_level=_mapping_float(data, "noise_level", 0.0),
            constraint_count=_mapping_int(data, "constraint_count", 0),
            gradient_available=_expect_bool("gradient_available", data.get("gradient_available", False)),
        )


@dataclass
class SearchStrategy:
    """Complete search-strategy recommendation chosen by the meta-controller."""

    optimizer: str
    mutation_rate: float
    population_size: int
    branching_factor: int
    exploration_exploitation: float
    compute_allocation: float
    convergence_threshold: float
    model_class: str

    def __post_init__(self) -> None:
        for name in ("optimizer", "model_class"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        self.mutation_rate = _validate_unit_interval("mutation_rate", self.mutation_rate)
        self.population_size = _validate_positive_int("population_size", self.population_size)
        self.branching_factor = _validate_positive_int("branching_factor", self.branching_factor)
        self.exploration_exploitation = _validate_unit_interval(
            "exploration_exploitation", self.exploration_exploitation
        )
        self.compute_allocation = _validate_positive_float("compute_allocation", self.compute_allocation)
        self.convergence_threshold = _validate_positive_float(
            "convergence_threshold", self.convergence_threshold
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize this strategy to a plain mapping."""
        return {
            "optimizer": self.optimizer,
            "mutation_rate": self.mutation_rate,
            "population_size": self.population_size,
            "branching_factor": self.branching_factor,
            "exploration_exploitation": self.exploration_exploitation,
            "compute_allocation": self.compute_allocation,
            "convergence_threshold": self.convergence_threshold,
            "model_class": self.model_class,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> SearchStrategy:
        """Build a `SearchStrategy` from a plain mapping."""
        return cls(
            optimizer=str(data["optimizer"]),
            mutation_rate=_mapping_float(data, "mutation_rate"),
            population_size=_mapping_int(data, "population_size"),
            branching_factor=_mapping_int(data, "branching_factor"),
            exploration_exploitation=_mapping_float(data, "exploration_exploitation"),
            compute_allocation=_mapping_float(data, "compute_allocation"),
            convergence_threshold=_mapping_float(data, "convergence_threshold"),
            model_class=str(data["model_class"]),
        )


@dataclass
class SearchOutcome:
    """One completed run stored as meta-learning evidence for later decisions."""

    problem: ProblemStructure
    strategy: SearchStrategy
    converged: bool
    generations_to_converge: int
    final_score: float
    record_id: str = field(default_factory=_next_outcome_id)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, ProblemStructure):
            raise TypeError("problem must be a ProblemStructure")
        if not isinstance(self.strategy, SearchStrategy):
            raise TypeError("strategy must be a SearchStrategy")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a boolean")
        self.generations_to_converge = _validate_non_negative_int(
            "generations_to_converge", self.generations_to_converge
        )
        self.final_score = float(self.final_score)
        if not np.isfinite(self.final_score):
            raise ValueError("final_score must be finite")
        if not isinstance(self.record_id, str):
            raise TypeError("record_id must be a string")
        if not self.record_id:
            raise ValueError("record_id must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        """Serialize this outcome to a plain mapping."""
        return {
            "problem": self.problem.to_mapping(),
            "strategy": self.strategy.to_mapping(),
            "converged": self.converged,
            "generations_to_converge": self.generations_to_converge,
            "final_score": self.final_score,
            "record_id": self.record_id,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> SearchOutcome:
        """Build a `SearchOutcome` from a plain mapping."""
        raw_problem = data["problem"]
        raw_strategy = data["strategy"]
        problem = (
            raw_problem
            if isinstance(raw_problem, ProblemStructure)
            else ProblemStructure.from_mapping(_expect_mapping("problem", raw_problem))
        )
        strategy = (
            raw_strategy
            if isinstance(raw_strategy, SearchStrategy)
            else SearchStrategy.from_mapping(_expect_mapping("strategy", raw_strategy))
        )
        return cls(
            problem=problem,
            strategy=strategy,
            converged=_expect_bool("converged", data["converged"]),
            generations_to_converge=_mapping_int(data, "generations_to_converge"),
            final_score=_mapping_float(data, "final_score"),
            record_id=str(data.get("record_id", _next_outcome_id())),
        )


def _expect_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


@dataclass
class SearchHistory:
    """Thin wrapper over past search outcomes plus optional `PatternMemory`.

    Callers can initialize this with:

    * concrete `SearchOutcome` objects,
    * plain outcome dictionaries matching `SearchOutcome.to_mapping()`, and/or
    * a shared `PatternMemory` instance that already contains meta-learning
      outcome payloads stored under this history's `intent`.
    """

    raw_records: InitVar[Iterable[SearchOutcome | Mapping[str, object]] | None] = None
    pattern_memory: PatternMemory | None = None
    intent: str = "meta-learning"
    records: list[SearchOutcome] = field(init=False, default_factory=list)

    def __post_init__(self, raw_records: Iterable[SearchOutcome | Mapping[str, object]] | None) -> None:
        if self.pattern_memory is not None and not isinstance(self.pattern_memory, PatternMemory):
            raise TypeError("pattern_memory must be a PatternMemory when provided")
        if not isinstance(self.intent, str):
            raise TypeError("intent must be a string")
        if not self.intent:
            raise ValueError("intent must be non-empty")
        self.records = []
        if raw_records is not None:
            for record in raw_records:
                self.records.append(self._coerce_record(record))

    def _coerce_record(self, record: SearchOutcome | Mapping[str, object]) -> SearchOutcome:
        if isinstance(record, SearchOutcome):
            return record
        if isinstance(record, Mapping):
            return SearchOutcome.from_mapping(record)
        raise TypeError("history records must be SearchOutcome objects or plain mappings")

    def append(self, record: SearchOutcome | Mapping[str, object]) -> SearchOutcome:
        """Append one outcome to the in-memory history."""
        outcome = self._coerce_record(record)
        self.records.append(outcome)
        return outcome

    def all_records(self) -> list[SearchOutcome]:
        """Return deduplicated local and `PatternMemory`-backed outcomes."""
        merged: dict[str, SearchOutcome] = {record.record_id: record for record in self.records}
        if self.pattern_memory is None:
            return list(merged.values())
        for pattern in self.pattern_memory.all_patterns(self.intent):
            try:
                outcome = SearchOutcome.from_mapping(_expect_mapping("payload", pattern.payload))
            except (KeyError, TypeError, ValueError):
                continue
            merged[outcome.record_id] = outcome
        return list(merged.values())


class MetaController:
    """Heuristic search-strategy selector for QES.

    The controller is deliberately transparent: it combines a problem feature
    vector, simple similarity scoring against prior outcomes, and weighted
    averaging / voting over historically successful strategies. This is
    meta-level adaptation over tracked history, not literal machine learning.
    """

    def __init__(self, *, min_similarity: float = 0.55) -> None:
        self.min_similarity = _validate_unit_interval("min_similarity", min_similarity)

    def recommend(self, problem: ProblemStructure, history: SearchHistory) -> SearchStrategy:
        """Recommend a full search strategy for `problem` using `history`."""
        if not isinstance(problem, ProblemStructure):
            raise TypeError("problem must be a ProblemStructure")
        if not isinstance(history, SearchHistory):
            raise TypeError("history must be a SearchHistory")

        defaults = self._defaults_for(problem)
        similar = self._similar_outcomes(problem, history.all_records())
        history_confidence = self._history_confidence(similar)
        historical_success = self._historical_success(similar)

        exploration = _clamp(
            defaults.exploration_exploitation
            + 0.20 * (1.0 - history_confidence)
            - 0.20 * history_confidence * historical_success
            + 0.10 * (1.0 - historical_success),
            0.05,
            0.95,
        )

        strategy = SearchStrategy(
            optimizer=defaults.optimizer,
            mutation_rate=defaults.mutation_rate,
            population_size=defaults.population_size,
            branching_factor=defaults.branching_factor,
            exploration_exploitation=exploration,
            compute_allocation=defaults.compute_allocation,
            convergence_threshold=defaults.convergence_threshold,
            model_class=defaults.model_class,
        )

        if similar:
            strategy = self._apply_history_bias(strategy, similar)

        branching_factor = self._branching_factor(
            problem=problem,
            strategy=strategy,
            history_confidence=history_confidence,
            historical_success=historical_success,
        )

        return SearchStrategy(
            optimizer=strategy.optimizer,
            mutation_rate=strategy.mutation_rate,
            population_size=strategy.population_size,
            branching_factor=branching_factor,
            exploration_exploitation=strategy.exploration_exploitation,
            compute_allocation=strategy.compute_allocation,
            convergence_threshold=strategy.convergence_threshold,
            model_class=strategy.model_class,
        )

    def _defaults_for(self, problem: ProblemStructure) -> SearchStrategy:
        dim_signal = problem.feature_vector()[0]
        width_signal = problem.feature_vector()[1]
        constraint_signal = problem.feature_vector()[4]

        smooth_enough = problem.ruggedness <= 0.55 and problem.noise_level <= 0.45
        gradient_preferred = problem.gradient_available and smooth_enough
        optimizer = "adaptive_gradient" if gradient_preferred else "evolutionary"
        model_class = "AdaptiveGradientSearch" if gradient_preferred else "EvolutionaryPopulation"

        mutation_rate = 0.06
        mutation_rate += 0.14 * dim_signal
        mutation_rate += 0.10 * width_signal
        mutation_rate += 0.12 * problem.ruggedness
        mutation_rate += 0.08 * problem.noise_level
        if gradient_preferred:
            mutation_rate -= 0.05
        mutation_rate = _clamp(mutation_rate, 0.01, 0.75)

        population_size = int(
            round(
                12
                + 48 * dim_signal
                + 28 * width_signal
                + 22 * problem.ruggedness
                + 10 * constraint_signal
            )
        )
        if optimizer == "evolutionary":
            population_size += 8
        population_size = int(_clamp(float(population_size), 8.0, 256.0))

        base_exploration = 0.15
        base_exploration += 0.18 * dim_signal
        base_exploration += 0.18 * width_signal
        base_exploration += 0.22 * problem.ruggedness
        base_exploration += 0.12 * problem.noise_level
        if not problem.gradient_available:
            base_exploration += 0.08
        base_exploration = _clamp(base_exploration, 0.05, 0.85)

        compute_allocation = 0.35
        compute_allocation += 0.25 * dim_signal
        compute_allocation += 0.15 * width_signal
        compute_allocation += 0.10 * problem.noise_level
        compute_allocation += 0.10 * constraint_signal
        compute_allocation = _clamp(compute_allocation, 0.1, 1.0)

        convergence_threshold = 5e-4
        convergence_threshold *= 1.0 + 2.5 * dim_signal + 2.0 * problem.noise_level + 1.5 * width_signal
        if gradient_preferred:
            convergence_threshold *= 0.8
        convergence_threshold = _clamp(convergence_threshold, 1e-5, 5e-2)

        return SearchStrategy(
            optimizer=optimizer,
            mutation_rate=mutation_rate,
            population_size=population_size,
            branching_factor=max(1, population_size // 12),
            exploration_exploitation=base_exploration,
            compute_allocation=compute_allocation,
            convergence_threshold=convergence_threshold,
            model_class=model_class,
        )

    def _similarity(self, left: ProblemStructure, right: ProblemStructure) -> float:
        weights = np.array([0.26, 0.17, 0.20, 0.15, 0.08, 0.14], dtype=float)
        diff = np.abs(left.feature_vector() - right.feature_vector())
        distance = float(np.average(diff, weights=weights))
        return _clamp(1.0 - distance, 0.0, 1.0)

    def _similar_outcomes(
        self, problem: ProblemStructure, records: Iterable[SearchOutcome]
    ) -> list[tuple[SearchOutcome, float, float]]:
        scored: list[tuple[SearchOutcome, float, float]] = []
        records_list = list(records)
        if not records_list:
            return scored

        raw_scores = np.array([record.final_score for record in records_list], dtype=float)
        min_score = float(np.min(raw_scores))
        max_score = float(np.max(raw_scores))

        for record in records_list:
            similarity = self._similarity(problem, record.problem)
            if similarity < self.min_similarity:
                continue
            if math.isclose(max_score, min_score):
                score_quality = 1.0
            else:
                score_quality = 1.0 - ((record.final_score - min_score) / (max_score - min_score))
            speed_quality = 1.0 / (1.0 + float(record.generations_to_converge))
            quality = 0.55 * (1.0 if record.converged else 0.0) + 0.30 * score_quality + 0.15 * speed_quality
            scored.append((record, similarity, quality))

        scored.sort(key=lambda item: item[1] * item[2], reverse=True)
        return scored[:8]

    def _history_confidence(self, similar: list[tuple[SearchOutcome, float, float]]) -> float:
        if not similar:
            return 0.0
        weight_sum = sum(similarity * quality for _, similarity, quality in similar)
        return _clamp(weight_sum / 2.5, 0.0, 1.0)

    def _historical_success(self, similar: list[tuple[SearchOutcome, float, float]]) -> float:
        if not similar:
            return 0.5
        weight_total = sum(similarity * max(quality, 1e-9) for _, similarity, quality in similar)
        if weight_total <= 0.0:
            return 0.5
        success_total = sum(
            similarity * max(quality, 1e-9) * (1.0 if record.converged else 0.0)
            for record, similarity, quality in similar
        )
        return _clamp(success_total / weight_total, 0.0, 1.0)

    def _apply_history_bias(
        self, base: SearchStrategy, similar: list[tuple[SearchOutcome, float, float]]
    ) -> SearchStrategy:
        weight_total = sum(similarity * quality for _, similarity, quality in similar)
        if weight_total <= 0.0:
            return base

        strength = _clamp(weight_total / (weight_total + 1.0), 0.0, 0.85)
        weighted_records = [(record, similarity * quality) for record, similarity, quality in similar]

        optimizer_weights: dict[str, float] = {}
        model_weights: dict[str, float] = {}
        for record, weight in weighted_records:
            optimizer_weights[record.strategy.optimizer] = (
                optimizer_weights.get(record.strategy.optimizer, 0.0) + weight
            )
            model_weights[record.strategy.model_class] = (
                model_weights.get(record.strategy.model_class, 0.0) + weight
            )

        optimizer = max(optimizer_weights, key=lambda name: optimizer_weights[name])
        model_class = max(model_weights, key=lambda name: model_weights[name])

        mutation_rate = self._weighted_average("mutation_rate", base.mutation_rate, weighted_records)
        population_size = int(
            round(self._weighted_average("population_size", float(base.population_size), weighted_records))
        )
        exploration = self._weighted_average(
            "exploration_exploitation", base.exploration_exploitation, weighted_records
        )
        compute_allocation = self._weighted_average(
            "compute_allocation", base.compute_allocation, weighted_records
        )
        convergence_threshold = self._weighted_average(
            "convergence_threshold", base.convergence_threshold, weighted_records
        )

        return SearchStrategy(
            optimizer=optimizer,
            mutation_rate=_clamp(
                (1.0 - strength) * base.mutation_rate + strength * mutation_rate, 0.01, 0.95
            ),
            population_size=max(
                4,
                int(round((1.0 - strength) * base.population_size + strength * population_size)),
            ),
            branching_factor=base.branching_factor,
            exploration_exploitation=_clamp(
                (1.0 - strength) * base.exploration_exploitation + strength * exploration, 0.05, 0.95
            ),
            compute_allocation=_clamp(
                (1.0 - strength) * base.compute_allocation + strength * compute_allocation, 0.05, 1.0
            ),
            convergence_threshold=_clamp(
                (1.0 - strength) * base.convergence_threshold + strength * convergence_threshold,
                1e-6,
                1.0,
            ),
            model_class=model_class,
        )

    def _weighted_average(
        self, attr_name: str, baseline: float, weighted_records: list[tuple[SearchOutcome, float]]
    ) -> float:
        if not weighted_records:
            return baseline
        numerator = sum(
            float(getattr(record.strategy, attr_name)) * weight for record, weight in weighted_records
        )
        denominator = sum(weight for _, weight in weighted_records)
        if denominator <= 0.0:
            return baseline
        return numerator / denominator

    def _branching_factor(
        self,
        *,
        problem: ProblemStructure,
        strategy: SearchStrategy,
        history_confidence: float,
        historical_success: float,
    ) -> int:
        dim_signal, width_signal, _, _, constraint_signal, _ = problem.feature_vector()
        uncertainty = _clamp(
            0.40 * strategy.exploration_exploitation
            + 0.20 * problem.noise_level
            + 0.20 * problem.ruggedness
            + 0.20 * dim_signal,
            0.0,
            1.0,
        )
        risk = _clamp(
            0.20
            + 0.35 * constraint_signal
            + 0.20 * problem.noise_level
            + 0.25 * (1.0 - strategy.compute_allocation),
            0.0,
            1.0,
        )
        divergence = _clamp(0.45 * width_signal + 0.35 * dim_signal + 0.20 * problem.ruggedness, 0.0, 1.0)
        novelty = _clamp(1.0 - history_confidence * historical_success, 0.0, 1.0)
        base = max(2, int(round(strategy.population_size / 12)))
        return adaptive_branch_count(
            uncertainty=uncertainty,
            risk=risk,
            divergence=divergence,
            compute_budget=_clamp(strategy.compute_allocation, 1e-6, 1.0),
            novelty=novelty,
            historical_success=historical_success,
            base=base,
            min_count=1,
            max_count=min(64, max(1, strategy.population_size)),
        )


def record_outcome(
    history: SearchHistory,
    problem: ProblemStructure,
    strategy: SearchStrategy,
    converged: bool,
    generations_to_converge: int,
    final_score: float,
) -> SearchOutcome:
    """Append one run outcome to `history` and optionally mirror it to `PatternMemory`."""
    if not isinstance(history, SearchHistory):
        raise TypeError("history must be a SearchHistory")
    outcome = SearchOutcome(
        problem=problem,
        strategy=strategy,
        converged=converged,
        generations_to_converge=generations_to_converge,
        final_score=final_score,
    )
    history.append(outcome)
    if history.pattern_memory is not None:
        history.pattern_memory.store(
            Pattern(
                intent=history.intent,
                context={
                    "dimensionality": problem.dimensionality,
                    "gradient_available": problem.gradient_available,
                },
                payload=outcome.to_mapping(),
                phi=0.0 if converged else 1.0,
                cci=max(0.0, float(final_score)),
                margin=1.0 / (1.0 + float(generations_to_converge)),
                id=outcome.record_id,
            )
        )
    return outcome


__all__ = [
    "MetaController",
    "ProblemStructure",
    "SearchHistory",
    "SearchOutcome",
    "SearchStrategy",
    "record_outcome",
]
