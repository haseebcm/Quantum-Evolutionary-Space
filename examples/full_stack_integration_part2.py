"""Full-stack integration demo, part 2: the remaining QES modules in one run.

This companion example picks up where `examples/full_stack_integration.py`
stops and wires together the still-unused modules in a single linear pass:

    1. qes.state_space       -- defines the bounded coordinate envelope.
    2. qes.domain            -- builds a domain-nullified metric/effective state.
    3. qes.dynamics          -- advances the room under drift + control.
    4. qes.divergence        -- measures DR/HSA health against the reference.
    5. qes.closure           -- checks viable-set distance, safe actions,
                                and cross-domain closure.
    6. qes.multi_reality     -- simulates branched trajectories, projects
                                nodes, replicates a pattern, and runs VQCE.
    7. qes.digital_twin      -- scores simulated rooms against an observed
                                target and updates belief weights.
    8. qes.convergence       -- summarizes how concentrated those weights are.
    9. qes.selection         -- filters candidate kinds and keeps survivors.
   10. qes.compute_allocator -- distributes CPU/GPU budget across candidates.
   11. qes.orchestrator      -- applies lifecycle transitions and ACROS
                                corrections.
   12. qes.expansion         -- births/routs/expands worlds inside a universe.
   13. qes.h11x             -- runs the six-layer engineering admissibility
                                stack and export barrier.
   14. qes.intelligence      -- performs adaptive optimization on the chosen
                                room.
   15. qes.multi_agent       -- spawns agents, assigns roles, routes a task,
                                and broadcasts it.
   16. qes.qsee              -- evolves isolated QEL streams and synchronizes
                                one deterministic output.
   17. qes.worker            -- runs one SQLite-backed worker-pool cycle.

Like part 1, the timing/memory numbers printed at the end are real
`time.perf_counter()` / `tracemalloc` measurements of ordinary Python and
NumPy work on this CPU/RAM. Nothing here is a literal parallel universe or
free compute source.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from qes.closure import (
    CrossDomainVerifier,
    PermissionClosure,
    action_selection,
    collapse_proximity,
    safe_exploration_closure,
)
from qes.closure import (
    divergence as closure_divergence,
)
from qes.closure import (
    divergence_gradient as closure_divergence_gradient,
)
from qes.compute_allocator import ComputeAllocator, RoomComputeProfile
from qes.convergence import (
    convergence_coefficient,
    gini_coefficient,
    kl_divergence,
    normalized_entropy,
)
from qes.digital_twin import DigitalTwin
from qes.divergence import DSA
from qes.domain import DomainNullification
from qes.dynamics import RoomDynamics
from qes.expansion import DomainBirthKernel, InfinityRouter, MetaExpansionEngine
from qes.h11x import H11X
from qes.intelligence import OptimizationResult, optimize
from qes.multi_agent import AutonomousNodeRouter, CognitiveAgentSpawner, LayeredRoleEngine
from qes.multi_reality import (
    VQCE,
    PatternReplicationLayer,
    SimulationEngine,
    VirtualNodeProjection,
)
from qes.orchestrator import Acros, AdaptiveAcros, RoomLifecycle
from qes.patterns import Pattern, PatternMemory
from qes.qsee import QSEE11L
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.selection import GenesisSelectionPipeline
from qes.state_space import MCCStateSpace
from qes.universe import Universe
from qes.worker import run as run_worker
from qes.world import World


@dataclass
class CandidateReality:
    """Selection/allocation record for one simulated room."""

    name: str
    kind: str
    room: Room
    twin_error: float
    posterior_weight: float


def format_vector(x: np.ndarray) -> str:
    """Compact array formatting for status lines."""
    return np.array2string(np.round(x, 3), separator=", ")


def print_module(name: str, detail: str) -> None:
    """One concise line per module."""
    print(f"[{name:<17}] {detail}")


def make_seed_room(space: MCCStateSpace) -> Room:
    """Construct the bounded room used across the walkthrough."""
    return Room(
        x=np.array([1.0, -0.7, 0.4]),
        x_star=space.reference.copy(),
        lower=space.lower.copy(),
        upper=space.upper.copy(),
        activation=np.array([1.0, 0.8, 0.5]),
    )


def candidate_score(candidate: CandidateReality) -> float:
    """Lower is better: small twin error, large posterior weight."""
    return candidate.twin_error - candidate.posterior_weight


def as_float(value: object) -> float:
    """Narrow an opaque scalar-like object to float for callback-heavy APIs."""
    return float(cast(float, value))


def combine_qsee_states(states: list[object]) -> dict[str, object]:
    """Produce one deterministic summary from the aligned QEL states."""
    active_states = [state for state in states if state is not None]
    scalar_sum = sum(
        float(state)
        for state in active_states
        if isinstance(state, (int, float)) and not isinstance(state, bool)
    )
    return {
        "active_streams": len(active_states),
        "scalar_sum": round(float(scalar_sum), 4),
    }


def run_worker_once(worker_state_dir: Path) -> int:
    """Run one worker cycle, then clean up its SQLite state."""
    worker_logger = logging.getLogger("qes.worker")
    previous_level = worker_logger.level
    worker_logger.setLevel(logging.WARNING)

    old_state_dir = os.environ.get("QES_RUNTIME_STATE_DIR")
    os.environ["QES_RUNTIME_STATE_DIR"] = str(worker_state_dir)
    worker_state_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_worker(worker_count=2, cycles=1, interval_seconds=0.0)
        db_path = worker_state_dir / "qes_runtime.sqlite"
        size = db_path.stat().st_size if db_path.exists() else 0
    finally:
        if old_state_dir is None:
            os.environ.pop("QES_RUNTIME_STATE_DIR", None)
        else:
            os.environ["QES_RUNTIME_STATE_DIR"] = old_state_dir
        worker_logger.setLevel(previous_level)
        shutil.rmtree(worker_state_dir, ignore_errors=True)

    return size


def main() -> None:
    rng = np.random.default_rng(23)
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES FULL-STACK INTEGRATION RUN (PART 2)")
    print("=" * 60)

    # --- 1. State-space envelope -----------------------------------------
    state_space = MCCStateSpace(
        lower=np.array([-1.5, -1.5, -1.5]),
        upper=np.array([1.5, 1.5, 1.5]),
        reference=np.array([0.0, 0.0, 0.0]),
        names=["position", "velocity", "bias"],
    )
    seed = make_seed_room(state_space)
    print_module(
        "state_space",
        f"dim={state_space.dim}, volume={state_space.volume():.1f}, "
        f"seed={state_space.labeled(seed.x)}",
    )

    # --- 2. Domain-nullified metric --------------------------------------
    domain = DomainNullification(
        [
            np.diag([1.0, 0.2, 0.1]),
            np.diag([0.3, 1.1, 0.2]),
            np.diag([0.1, 0.4, 1.5]),
        ]
    )
    metric = domain.metric(seed.activation)
    effective = DomainNullification.effective_state(seed.activation, seed.x)
    print_module(
        "domain",
        f"effective={format_vector(effective)}, trace(W)={float(np.trace(metric)):.3f}",
    )

    # --- 3. Dynamics ------------------------------------------------------
    dynamics = RoomDynamics(
        drift=lambda x, t: -0.25 * x,
        control_matrix=np.eye(seed.dim) * 0.5,
    )
    control = np.array([-0.10, 0.05, 0.00])
    x_next = dynamics.step_stochastic(
        seed.x,
        t=0.0,
        dt=0.5,
        sigma=0.0,
        u=control,
        rng=rng,
    )
    evolved_room = seed.clone(x=x_next)
    print_module(
        "dynamics",
        f"x_next={format_vector(x_next)}, admissible={state_space.is_admissible(x_next)}",
    )

    # --- 4. Divergence / health ------------------------------------------
    dsa = DSA()
    divergence_result = dsa.update(
        x=evolved_room.x,
        x_star=evolved_room.x_star,
        dt=0.5,
        w=metric,
        k=1.3,
        baseline=0.3,
    )
    print_module(
        "divergence",
        f"DR={divergence_result.dr:.4f}, health={divergence_result.health:.4f}, "
        f"state={divergence_result.state()}",
    )

    # --- 5. Closure / safe action selection ------------------------------
    viable_set = [
        state_space.reference.copy(),
        np.array([0.2, -0.1, 0.05]),
        np.array([0.0, 0.0, 0.2]),
    ]
    closure_distance = closure_divergence(evolved_room.x, viable_set)
    closure_grad = closure_divergence_gradient(evolved_room.x, viable_set)
    closure_score = collapse_proximity(
        closure_distance,
        float(np.linalg.norm(closure_grad)),
        divergence_result.dr,
    )

    permission = PermissionClosure(
        candidate_actions=[-0.2, 0.0, 0.15, 0.35],
        constraint_fn=lambda _x, u: abs(as_float(u)) - 0.25,
    ).evaluate(x=None)
    chosen_action = as_float(
        action_selection(
            x=float(np.linalg.norm(evolved_room.x)),
            actions=permission.admissible_actions,
            j_fn=lambda x, u: abs(as_float(x) + as_float(u)),
            s_fn=lambda next_state: abs(as_float(next_state) - 0.25),
            transition_fn=lambda x, u: as_float(x) + as_float(u),
            lam=0.5,
        )
    )
    safe_closure = safe_exploration_closure(
        policies=[0.05, 0.15, 0.20],
        search_operator=lambda p: as_float(p) + 0.02,
        is_admissible_fn=lambda p: as_float(p) < 0.30,
    )
    control_vector = np.array([chosen_action, -0.5 * chosen_action, 0.0])
    verifier = CrossDomainVerifier(
        encoder=lambda y: np.asarray(y, dtype=float),
        decoder=lambda x: x,
        transition_fn=lambda x, u, xi: x + np.asarray(u, dtype=float) + np.asarray(xi, dtype=float),
        state_space_check=lambda x: state_space.is_admissible(x),
        constraint_fn=lambda _x, u: float(np.max(np.abs(np.asarray(u, dtype=float))) - 0.4),
    )
    verification = verifier.verify(
        y=evolved_room.x,
        u=control_vector,
        xi=np.zeros(seed.dim),
    )
    print_module(
        "closure",
        f"D(x)={closure_distance:.4f}, action={chosen_action:.2f}, "
        f"safe_map={safe_closure}, state_closed={verification.state_closed}",
    )

    # --- 6. Multi-reality fabric -----------------------------------------
    simulation_engine = SimulationEngine()
    step_fns = [
        lambda room, t, dt: dynamics.integrate_rk4(
            room.x,
            u=np.array([-0.05, 0.02, 0.00]),
            t=t,
            dt=dt,
        ),
        lambda room, t, dt: dynamics.step_stochastic(
            room.x,
            t=t,
            dt=dt,
            sigma=0.0,
            u=np.array([0.02, 0.00, -0.03]),
            rng=rng,
        ),
        lambda room, t, dt: dynamics.step_discrete(
            room.x,
            u=np.array([0.00, 0.05, 0.05]),
            xi=np.zeros(room.dim),
            t=t,
        ),
    ]
    simulated_rooms = simulation_engine.simulate(evolved_room, step_fns, steps=4, dt=0.5)

    projection = VirtualNodeProjection(generator=RealityGenerator(rng=np.random.default_rng(24)))
    nodes = projection.project(seed, count=3, scale=0.08)
    pattern_memory = PatternMemory()
    pattern = Pattern(
        intent="part2-branch-seed",
        context={"phase": "branch"},
        payload=np.round(evolved_room.x, 3).tolist(),
        phi=float(divergence_result.dr),
        cci=float(closure_score),
        margin=1.0 / (1.0 + divergence_result.dr),
    )
    replication = PatternReplicationLayer(memory=pattern_memory)
    replicated = replication.replicate(pattern, nodes)
    recalled = replication.best_for("part2-branch-seed", {"phase": "branch"})

    vqce = VQCE(h11=1.2, stability_tolerance=1.0)
    state_key = vqce.allocate_state(dim=seed.dim)
    vqce_result = vqce.compute(state_key, simulated_rooms[0].x, compute_fn=lambda x: x * 1.2)
    print_module(
        "multi_reality",
        f"simulated={len(simulated_rooms)}, nodes={len(nodes)}, "
        f"replicated={replicated}, best_pattern={recalled.id if recalled else None}, "
        f"vqce_validated={vqce_result.validated}",
    )

    # --- 7. Digital twin weighting ---------------------------------------
    observed = np.array([0.15, -0.05, 0.10])
    twin = DigitalTwin(noise_covariance=np.eye(seed.dim) * 0.25, window=5)
    likelihood_cov = np.eye(seed.dim) * 0.25
    prior = np.array([0.34, 0.33, 0.33], dtype=float)

    deltas: list[float] = []
    likelihoods: list[float] = []
    candidate_names = ["guided-rk4", "guided-drift", "explore-kick"]
    candidate_kinds = ["guided", "guided", "explore"]

    for room in simulated_rooms:
        residual = twin.twin_residual(observed=observed, predicted=room.x)
        delta = twin.weighted_twin_error(residual)
        twin.record_residual(delta)
        deltas.append(delta)
        likelihoods.append(DigitalTwin.likelihood_gaussian(residual, likelihood_cov))

    posterior = DigitalTwin.evidence_update(prior.tolist(), likelihoods)
    drift_flag = twin.detect_drift(threshold=1.0)
    print_module(
        "digital_twin",
        f"posterior={format_vector(posterior)}, mean_delta={twin.rolling_mean_error():.4f}, "
        f"drift={drift_flag}",
    )

    # --- 8. Convergence summary ------------------------------------------
    posterior_list = posterior.tolist()
    convergence = convergence_coefficient(posterior_list)
    entropy = normalized_entropy(posterior_list)
    kl = kl_divergence(posterior_list, [1.0 / posterior.size] * posterior.size)
    gini = gini_coefficient(posterior_list)
    print_module(
        "convergence",
        f"Cq={convergence:.4f}, Hbar={entropy:.4f}, KL(uniform)={kl:.4f}, gini={gini:.4f}",
    )

    # --- 9. Selection pipeline -------------------------------------------
    candidates = [
        CandidateReality(
            name=name,
            kind=kind,
            room=room,
            twin_error=delta,
            posterior_weight=float(weight),
        )
        for name, kind, room, delta, weight in zip(
            candidate_names,
            candidate_kinds,
            simulated_rooms,
            deltas,
            posterior_list,
            strict=True,
        )
    ]
    error_limit = float(np.quantile(np.asarray(deltas, dtype=float), 0.75))

    def selection_gate_admissible(candidate: object) -> float:
        reality = cast(CandidateReality, candidate)
        return 0.0 if state_space.is_admissible(reality.room.x) else 1.0

    def selection_gate_error(candidate: object) -> float:
        reality = cast(CandidateReality, candidate)
        return reality.twin_error - error_limit

    def selection_gate_weight(candidate: object) -> float:
        reality = cast(CandidateReality, candidate)
        return 0.05 - reality.posterior_weight

    def selection_signature(candidate: object) -> object:
        return cast(CandidateReality, candidate).kind

    def selection_score(candidate: object) -> float:
        return candidate_score(cast(CandidateReality, candidate))

    pipeline = GenesisSelectionPipeline(
        gates=[selection_gate_admissible, selection_gate_error, selection_gate_weight],
        signature_fn=selection_signature,
        score_fn=selection_score,
    )
    pipeline_result = pipeline.run(candidates)
    survivors = [
        trace.survivor
        for trace in pipeline_result.kinds.values()
        if isinstance(trace.survivor, CandidateReality)
    ]
    winner = min(survivors, key=candidate_score)
    survivor_names = [candidate.name for candidate in survivors]
    print_module(
        "selection",
        f"history={pipeline_result.population_history}, survivors={survivor_names}, "
        f"winner={winner.name}",
    )

    # --- 10. Compute allocation ------------------------------------------
    allocator = ComputeAllocator()
    max_delta = max(deltas)
    profiles = [
        RoomComputeProfile(
            permission=max(0.05, 1.0 - candidate.twin_error / (max_delta + 1e-9)),
            uncertainty=max(0.05, 1.0 - candidate.posterior_weight),
            risk=candidate.twin_error,
            value=max(0.05, candidate.posterior_weight),
        )
        for candidate in candidates
    ]
    cpu_budget = allocator.allocate_with_floor(profiles, total=120.0, min_share=0.2)
    resources = allocator.allocate_multi_resource(
        profiles,
        capacities={"cpu": 120.0, "gpu": 12.0},
    )
    for candidate, cpu in zip(candidates, cpu_budget, strict=True):
        candidate.room.compute["cpu"] = float(cpu)
    print_module(
        "compute_allocator",
        f"cpu={format_vector(cpu_budget)}, gpu={format_vector(resources['gpu'])}, "
        f"fits={allocator.check_resource_constraint(resources['cpu'], 120.0)}",
    )

    # --- 11. Orchestration / lifecycle -----------------------------------
    companion = max(candidates, key=lambda candidate: candidate.posterior_weight)
    RoomLifecycle.transition(winner.room, "Active")
    if companion.room.state == "Seed":
        RoomLifecycle.transition(companion.room, "Active")
    merged_room = RoomLifecycle.merge(winner.room, companion.room)

    acros = Acros(
        gradient_fn=lambda x, context: x - np.asarray(context["reference"], dtype=float),
        gain=0.3,
    )
    derivative = acros.state_derivative(
        winner.room.x,
        t=0.0,
        drift_fn=lambda x, t: -0.1 * x,
        chi=1.0,
        context={"reference": state_space.reference},
    )
    adaptive_acros = AdaptiveAcros(
        gradient_fn=lambda x, context: x - np.asarray(context["reference"], dtype=float),
        gain=0.5,
    )
    adaptive_correction = adaptive_acros.correction(
        winner.room.x,
        chi=1.0,
        context={"reference": state_space.reference},
    )
    print_module(
        "orchestrator",
        f"winner_state={winner.room.state}, merged_state={merged_room.state}, "
        f"|xdot|={float(np.linalg.norm(derivative)):.4f}, "
        f"|adapt|={float(np.linalg.norm(adaptive_correction)):.4f}",
    )

    # --- 12. Expansion ----------------------------------------------------
    universe = Universe(dt=0.5, name="part2-universe")
    primary_world = World(name="primary-domain")
    primary_world.fields["temperature"] = 42.0
    universe.add_world(primary_world)

    birth_kernel = DomainBirthKernel()
    shadow_world = birth_kernel.spawn_from(primary_world, name="shadow-domain")
    universe.add_world(shadow_world)

    router = InfinityRouter(kernel=birth_kernel)
    routed_world = router.route(
        "analysis",
        [primary_world, shadow_world],
        capacity_fn=lambda world: world.name == "shadow-domain",
    )
    expansion = MetaExpansionEngine(kernel=birth_kernel)
    overflow_world = expansion.expand(
        universe,
        load_fn=lambda world: 1.2 if world.name in {"primary-domain", "shadow-domain"} else 0.0,
        threshold=1.0,
        name="overflow-domain",
    )
    print_module(
        "expansion",
        f"routed={routed_world.name}, expanded={overflow_world.name if overflow_world else None}, "
        f"worlds={len(universe.worlds)}",
    )

    # --- 13. H11X engineering admissibility ------------------------------
    h11x = H11X()
    h11x_result = h11x.evaluate(
        n=0.0,
        stress=float(np.linalg.norm(merged_room.x)),
        load_paths=float(np.linalg.norm(derivative)),
        energy_flow=float(np.linalg.norm(adaptive_correction)),
        temporal_stability=1.0 + convergence,
        domain_checks={
            "bounds": state_space.is_admissible(merged_room.x),
            "health": divergence_result.health < 5.0,
        },
        push_to_failure_fn=lambda: 200.0,
        recovery_check_fn=lambda value: value < 100.0,
        internal_state={"x": merged_room.x.copy()},
        derive_fn=lambda state: np.round(np.asarray(state["x"], dtype=float), 3).tolist(),
        reintegrate_fn=lambda state: {
            "x": np.clip(
                np.asarray(state["x"], dtype=float) * 0.8,
                merged_room.lower,
                merged_room.upper,
            )
        },
        relax_fn=lambda state: {"x": np.asarray(state["x"], dtype=float) * 0.9},
        reform_fn=lambda state: {"x": np.asarray(state["x"], dtype=float) + 0.02},
    )
    geometry_severity = (
        h11x_result.geometry.severity() if h11x_result.geometry is not None else float("nan")
    )
    print_module(
        "h11x",
        f"admitted={h11x_result.admitted}, corrected={h11x_result.correction is not None}, "
        f"severity={geometry_severity:.4f}",
    )

    # --- 14. Adaptive optimization ---------------------------------------
    target = np.array([0.15, -0.05, 0.10])
    optimization_seed = merged_room.clone(memory={})
    optimization: OptimizationResult = optimize(
        lambda x: float(np.sum((x - target) ** 2)),
        optimization_seed,
        iterations=40,
        population=12,
        branch_scale=0.12,
        permission_theta=4.0,
        rng=np.random.default_rng(25),
    )
    print_module(
        "intelligence",
        f"best_value={optimization.best_value:.6f}, survivors={optimization.survivors}, "
        f"evaluations={optimization.evaluations}",
    )

    # --- 15. Multi-agent coordination ------------------------------------
    spawner = CognitiveAgentSpawner()
    agents = spawner.spawn_population(
        4,
        name_prefix="solver",
        state_fn=lambda index: {"belief": float(posterior[index % posterior.size])},
    )
    roles = LayeredRoleEngine(layers=[["coordinator"], ["analyst", "worker"]])
    roles.assign(agents)
    agent_router = AutonomousNodeRouter()
    selected_agent = agent_router.route(
        {"winner": winner.name},
        agents,
        fitness_fn=lambda agent, task: float(agent.state.get("belief", 0.0))
        + (0.5 if agent.memory.get("role") == "coordinator" else 0.0),
    )
    delivered = agent_router.broadcast_task(
        {"best_x": np.round(optimization.best_x, 3).tolist()},
        agents,
        t=1.0,
    )
    received = sum(len(agent.receive()) for agent in agents)
    print_module(
        "multi_agent",
        f"selected={selected_agent.name if selected_agent else None}, "
        f"coordinators={len(roles.agents_with_role('coordinator'))}, "
        f"broadcast={delivered}/{received}",
    )

    # --- 16. QSEE-11L streams --------------------------------------------
    qsee = QSEE11L()
    qsee.evolve(
        {
            1: lambda state: float(np.linalg.norm(optimization.best_x)),
            2: lambda state: round(convergence, 4),
            6: lambda state: [candidate.name for candidate in survivors],
            11: lambda state: {"winner": winner.name, "survivors": optimization.survivors},
        }
    )
    qsee_output = qsee.request_output(combine_fn=combine_qsee_states)
    print_module(
        "qsee",
        f"active={qsee_output.output['active_streams']}, "
        f"scalar_sum={qsee_output.output['scalar_sum']}",
    )

    # --- 17. Worker service ----------------------------------------------
    worker_state_dir = Path(__file__).resolve().parent / "_part2_worker_state"
    worker_db_size = run_worker_once(worker_state_dir)
    print_module("worker", f"ran one cycle, sqlite_bytes={worker_db_size}")

    elapsed = time.perf_counter() - t_start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print(
        "  modules touched      : closure, compute_allocator, convergence, "
        "digital_twin, divergence, domain, dynamics, expansion, h11x, "
        "intelligence, multi_agent, multi_reality, orchestrator, qsee, "
        "selection, state_space, worker (17 modules)"
    )
    print(
        "  This is still a classical NumPy/Python optimization pipeline. "
        "The 'universes', 'worlds', and 'streams' above are data structures "
        "running on this machine's CPU and RAM, with ordinary computing cost."
    )


if __name__ == "__main__":
    main()
