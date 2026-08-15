"""Phase 2 -- the complete closed-loop autonomous cycle.

    INTENT -> GENERATION -> REALITY POPULATION -> EXECUTION -> DIVERGENCE
        -> PERMISSION -> SELECTION -> CONVERGENCE -> KNOWLEDGE EXTRACTION
        -> PATTERN MEMORY -> ADAPTIVE REGENERATION -> (new population) -> ...

`QESSpace.step()` already implements the inner
``E -> D -> P -> S -> C`` (execute/measure-divergence/check-permission/
select/converge) master operator. What was missing to make QES
*genuinely* closed-loop was the outer ring: a named intent, extracting
knowledge from each generation's outcome, saving it to `PatternMemory`,
and using both the outcome and the memory to decide how the *next*
population is generated -- rather than a human re-seeding it by hand
every time.

`AutonomousCycle` wires this outer ring around an existing `QESSpace`:
each call to `run_generation()` ticks the space forward, extracts the
dominant room's outcome as a `Pattern`, computes an adaptive number of
children for the next generation via `adaptive_branch_count()`, branches
them with a `RealityGenerator`, and records `SPAWN`/`EXECUTE`/`DIVERGE`/
`PERMISSION`/`SELECT`/`MERGE` events plus a full `InvariantEngine` check
into an `EventLog` -- so every generation is both self-driving and fully
auditable/replayable, not just self-driving.

This is still an ordinary classical loop: each generation costs real
CPU/RAM (see `examples/full_stack_integration.py` for measured numbers).
"Closed-loop" here means the population regenerates itself from its own
outcomes without a human manually re-seeding it -- not that the loop
runs for free.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qes.events import EventLog
from qes.invariants import InvariantEngine, InvariantReport
from qes.patterns import Pattern, PatternMemory
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace, SpaceTelemetry

ScoreFn = Callable[[Room], float]


def adaptive_branch_count(
    uncertainty: float,
    risk: float,
    divergence: float,
    compute_budget: float,
    novelty: float,
    historical_success: float,
    *,
    base: int = 4,
    min_count: int = 1,
    max_count: int = 64,
) -> int:
    """N_children = f(uncertainty, risk, divergence, compute, novelty, historical_success).

    A documented heuristic (not a literal physical law): branch *more*
    when the search is uncertain, exploring novel territory, or has a
    poor track record for this intent so far (worth exploring); branch
    *less* when risk or divergence from target are already high, or the
    compute budget is tight (worth conserving resources).

    Args:
        uncertainty: in [0, 1], how unresolved the current search is.
        risk: in [0, 1], how costly a bad candidate would be.
        divergence: in [0, 1], normalized current distance from target.
        compute_budget: in (0, 1], fraction of the nominal per-generation
            compute allowance available right now (throttles all output).
        novelty: in [0, 1], how unexplored the current region is.
        historical_success: in [0, 1], this intent's past success rate
            (from `PatternMemory`); low values push toward more
            exploration, high values push toward exploitation.
        base: nominal branch count at neutral (0.5) inputs.
        min_count: floor on the returned count.
        max_count: ceiling on the returned count.

    Returns:
        An integer child count in ``[min_count, max_count]``.
    """
    for name, value in (
        ("uncertainty", uncertainty),
        ("risk", risk),
        ("divergence", divergence),
        ("novelty", novelty),
        ("historical_success", historical_success),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    if not 0.0 < compute_budget <= 1.0:
        raise ValueError(f"compute_budget must be in (0, 1], got {compute_budget!r}")
    if base <= 0:
        raise ValueError("base must be positive")
    if min_count < 0 or max_count < min_count:
        raise ValueError("require 0 <= min_count <= max_count")

    explore = 0.5 * uncertainty + 0.3 * novelty + 0.2 * (1.0 - historical_success)
    conserve = 0.5 * risk + 0.5 * divergence
    raw = base * (1.0 + 2.0 * explore - 1.0 * conserve) * compute_budget
    return int(max(min_count, min(max_count, round(raw))))


@dataclass
class GenerationReport:
    """Outcome of one full turn of the closed loop."""

    generation: int
    telemetry: SpaceTelemetry
    invariants: InvariantReport
    dominant_room_id: str | None
    pattern: Pattern | None
    n_children_next: int
    events_recorded: int


@dataclass
class CycleHistory:
    """Accumulated per-generation reports plus the shared event/lineage/memory state."""

    reports: list[GenerationReport] = field(default_factory=list)

    def convergence_series(self) -> list[float]:
        return [r.telemetry.convergence for r in self.reports]

    def entropy_series(self) -> list[float]:
        return [r.telemetry.entropy for r in self.reports]


class AutonomousCycle:
    """Drives `QESSpace` through the full closed autonomous cycle.

    Owns (or is handed) an `EventLog` and `InvariantEngine` so every
    generation is recorded and checked, and a `PatternMemory` so
    knowledge from past generations informs the *next* population size
    and starting point, closing the loop:
    generate -> execute -> measure -> permit -> select -> converge ->
    extract knowledge -> remember -> regenerate.
    """

    def __init__(
        self,
        intent: str,
        space: QESSpace,
        generator: RealityGenerator,
        score_fn: ScoreFn,
        memory: PatternMemory | None = None,
        events: EventLog | None = None,
        invariant_engine: InvariantEngine | None = None,
    ):
        if not isinstance(intent, str) or not intent:
            raise ValueError("intent must be a non-empty string")
        self.intent = intent
        self.space = space
        self.generator = generator
        self.score_fn = score_fn
        self.memory = memory if memory is not None else PatternMemory()
        self.events = events if events is not None else EventLog()
        self.invariant_engine = invariant_engine if invariant_engine is not None else InvariantEngine()
        self.generation = 0
        self.history = CycleHistory()

    # ------------------------------------------------------------------
    # INTENT + initial REALITY POPULATION
    # ------------------------------------------------------------------
    def seed(self, rooms: list[Room]) -> None:
        """Register the initial candidate population (generation 0)."""
        for room in rooms:
            self.space.add_room(room)
        self.events.record(
            "SPAWN",
            payload={"intent": self.intent, "count": len(rooms), "generation": self.generation},
            parent_ids=[],
            timestamp=self.space.time,
        )

    # ------------------------------------------------------------------
    # One full turn of the loop
    # ------------------------------------------------------------------
    def run_generation(
        self,
        ticks: int = 1,
        *,
        compute_budget: float = 1.0,
        risk: float = 0.3,
        novelty: float = 0.5,
    ) -> GenerationReport:
        """EXECUTION -> DIVERGENCE -> PERMISSION -> SELECTION -> CONVERGENCE
        (via `QESSpace.step()`) followed by KNOWLEDGE EXTRACTION -> PATTERN
        MEMORY -> ADAPTIVE REGENERATION."""
        events_before = len(self.events)
        telemetry = None
        for _ in range(max(1, ticks)):
            telemetry = self.space.step()
        assert telemetry is not None  # ticks >= 1 guaranteed above

        self.events.record(
            "EXECUTE",
            payload={"generation": self.generation, "ticks": ticks, "time": telemetry.time},
            timestamp=telemetry.time,
        )
        self.events.record(
            "DIVERGE",
            payload={"generation": self.generation, "entropy": telemetry.entropy},
            timestamp=telemetry.time,
        )
        self.events.record(
            "PERMISSION",
            payload={"generation": self.generation, "collapsed": telemetry.collapsed},
            timestamp=telemetry.time,
        )
        self.events.record(
            "SELECT",
            payload={"generation": self.generation, "active": telemetry.active},
            timestamp=telemetry.time,
        )

        # --- Invariant check across the whole space, every generation ---
        invariant_report = self.invariant_engine.check_space(self.space)

        # --- Knowledge extraction + pattern memory ---
        dominant = self.space.dominant_room()
        pattern: Pattern | None = None
        historical_success = 0.5
        if dominant is not None:
            prior = self.memory.generate(self.intent, {"generation": self.generation})
            if prior is not None:
                historical_success = max(0.0, min(1.0, 1.0 - prior.phi))
            score = self.score_fn(dominant)
            pattern = Pattern(
                intent=self.intent,
                context={"generation": self.generation},
                payload=dominant.x.tolist(),
                phi=max(0.0, 1.0 - telemetry.convergence),
                cci=float(score) if score >= 0 else 0.0,
                margin=telemetry.dominant_permission,
            )
            self.memory.store(pattern)
            self.memory.retire_dominated(self.intent)

        # --- Adaptive regeneration: branch the next population ---
        n_children = adaptive_branch_count(
            uncertainty=max(0.0, min(1.0, 1.0 - telemetry.convergence)),
            risk=risk,
            divergence=max(0.0, min(1.0, telemetry.entropy)),
            compute_budget=compute_budget,
            novelty=novelty,
            historical_success=historical_success,
        )
        if dominant is not None and n_children > 0:
            children = self.generator.branch(dominant, count=n_children)
            self.space.spawn(children)
            self.events.record(
                "MERGE",
                payload={"generation": self.generation, "n_children": n_children},
                parent_ids=[dominant.id],
                timestamp=telemetry.time,
            )

        report = GenerationReport(
            generation=self.generation,
            telemetry=telemetry,
            invariants=invariant_report,
            dominant_room_id=dominant.id if dominant else None,
            pattern=pattern,
            n_children_next=n_children,
            events_recorded=len(self.events) - events_before,
        )
        self.history.reports.append(report)
        self.generation += 1
        return report

    def run(
        self,
        generations: int,
        ticks_per_generation: int = 1,
        **kwargs: Any,
    ) -> list[GenerationReport]:
        """Run the closed loop for `generations` full turns."""
        return [
            self.run_generation(ticks_per_generation, **kwargs) for _ in range(generations)
        ]
