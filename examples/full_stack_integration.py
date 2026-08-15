"""Full-stack integration demo: every layer of QES wired into one run.

This is the single script to point to when asking "do all the modules
actually work together?" It chains, in order:

    1. qes.kernel            -- Digital Existence Kernel gates a candidate
                                 state into existence (Layers 1-5).
    2. qes.hypervisor        -- CosmicVP provisions the virtual resources
                                 (vm/network/storage) the run will use.
    3. qes.permission        -- Genesis admissibility gate for room seeds.
    4. qes.reality_generator -- branches seed rooms into a population.
    5. qes.space / qes.room  -- QESSpace evolves the room population.
    6. qes.agent / qes.world
       / qes.universe        -- worlds + a monitoring agent inside a
                                 top-level Universe, ticked forward in time.
    7. qes.runtime           -- a DistributedRuntime schedules each world's
                                 step as a prioritized job across worker
                                 threads (this is where real wall-clock
                                 time and thread scheduling actually happen).
    8. qes.validation        -- ValidationFramework + FunctionalSafetySystem
                                 + FailureRecoveryLoop check and, if needed,
                                 recover the resulting states.
    9. qes.equation_forge
       / qes.x_engine        -- x_engine_pipeline() perfects an equation and
                                 runs it through Apex-I execution + GeoM
                                 geometry checks.
   10. qes.patterns          -- the winning configuration is stored as a
                                 reusable Pattern for future runs.

Every step below is measured with `time.perf_counter()` and
`tracemalloc` and the real numbers are printed at the end -- this is
ordinary CPU/memory cost, not "free" computation, and the "universe"
here is a `Universe` *object* (a Python data structure), not a physical
reality. See docs/architecture-overview.md for the honest framing.

Run with:  python examples/full_stack_integration.py
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.agent import Agent
from qes.equation_forge import EquationForge
from qes.hypervisor import CosmicVP
from qes.kernel import ExistenceKernel
from qes.patterns import Pattern, PatternMemory
from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.runtime import DistributedRuntime
from qes.space import QESSpace
from qes.universe import Universe
from qes.validation import FailureRecoveryLoop, FunctionalSafetySystem, ValidationFramework
from qes.world import World
from qes.x_engine import GeoM, x_engine_pipeline


def make_seed_room(dim: int = 3) -> Room:
    return Room(
        x=np.zeros(dim),
        x_star=np.ones(dim) * 0.5,
        lower=-np.ones(dim),
        upper=np.ones(dim),
        activation=np.ones(dim),
    )


def mean_reverting_step(room: Room, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    pull = -0.2 * (room.x - room.x_star)
    noise = rng.normal(0.0, 0.02, size=room.dim)
    return room.x + dt * pull + noise


def monitor_policy(agent: Agent, observation: dict, t: float) -> str:
    agent.memory["last_tick"] = t
    agent.memory["ticks_seen"] = agent.memory.get("ticks_seen", 0) + 1
    return f"observed tick {t:.1f}"


def main() -> None:  # noqa: C901 - single linear walkthrough is the point
    rng = np.random.default_rng(7)
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES FULL-STACK INTEGRATION RUN")
    print("=" * 60)

    # --- 1. Digital Existence Kernel: gate the seed state into existence ---
    kernel = ExistenceKernel()
    kernel.null_origin(dim=3)  # Layer 1: s_0 = empty (documented, not instantiated)
    candidate = kernel.allow(np.zeros(3))  # Layer 2: empty -> S
    construction = kernel.construct(complexity_t=0.0, delta=0.4)  # Layer 3
    op_result = kernel.operate(candidate, lambda s: s + 0.0005)  # Layer 4+5
    print(f"[kernel]      allowed candidate, complexity={construction.complexity:.3f}, "
          f"identity_preserved={op_result.identity_preserved}")

    # --- 2. Hypervisor: provision the virtual resources this run will use ---
    hypervisor = CosmicVP(h11=1.0, capacity=16)
    vm = hypervisor.create_vm(spec={"role": "world-runner"})
    net = hypervisor.create_network(spec={"role": "telemetry-bus"})
    print(f"[hypervisor]  provisioned resources: {vm.id}, {net.id} "
          f"(nodes={hypervisor.nodes}, in-use={len(hypervisor.resources)}/"
          f"{hypervisor.nodes * hypervisor.capacity})")

    # --- 3-6. Universe: worlds + agent, evolved via QESSpace ---
    seed = make_seed_room()
    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=lambda room, t, dt: mean_reverting_step(room, t, dt, rng),
        dt=1.0,
    )
    space.spawn(RealityGenerator(rng=rng).branch(seed, count=25, scale=0.2))
    world = World(name="design-space", space=space)
    world.add_agent(Agent(name="monitor", policy=monitor_policy))

    universe = Universe(dt=1.0)
    universe.add_world(world)

    # --- 7. Runtime: schedule the universe's ticks as prioritized jobs ---
    runtime = DistributedRuntime(worker_count=2)
    for i in range(5):
        runtime.submit(f"universe-tick-{i}", lambda: universe.step(event="tick"), priority=1)
    jobs = runtime.run_all()
    status = runtime.status()
    print(f"[runtime]     ran {len(jobs)} scheduled ticks across "
          f"{status['workers']} worker threads, success_rate={status['success_rate']:.2f}")

    telemetry = universe.telemetry()
    monitor = world.agents[next(iter(world.agents))]
    print(f"[universe]    time={telemetry.time}, worlds={telemetry.world_count}, "
          f"agents={telemetry.agent_count}, monitor_memory={monitor.memory}")

    # --- 8. Validation: check + recover the final room population ---
    final_states = [room.x for room in space.rooms.values()]
    lower, upper = seed.lower, seed.upper
    safety = FunctionalSafetySystem(lower=lower, upper=upper, margin=0.05)
    framework = ValidationFramework()
    framework.add_check("within_bounds", lambda x: bool(np.all(x >= lower) and np.all(x <= upper)))
    framework.add_check("finite", lambda x: bool(np.all(np.isfinite(x))))
    reports = [framework.validate(x) for x in final_states]
    n_passed = sum(1 for r in reports if r.passed)
    avg_safety = float(np.mean([safety.safety_score(x) for x in final_states]))

    # Deliberately fail one state on purpose to exercise the Omega-Loop.
    recovery = FailureRecoveryLoop(lower=lower, upper=upper)
    broken_state = upper * 10.0  # out of bounds by construction
    recovered = recovery.recover(broken_state, regenerate_fn=lambda base: base + 0.01)
    print(f"[validation]  {n_passed}/{len(reports)} rooms passed all checks, "
          f"avg_safety_score={avg_safety:.3f}")
    print(f"[recovery]    Omega-Loop recovered an out-of-bounds state: "
          f"was_null={recovered.was_null}, recovered={recovered.recovered_state}")

    # --- 9. X-Engine: perfect an equation and execute it through GeoM ---
    def score_fn(equation) -> float:
        return float(np.sum(np.abs(list(equation.theta.values()))))

    forge = EquationForge(rng=rng)
    pipeline_result = x_engine_pipeline(
        intent="design-refinement",
        x=final_states[0],
        lower=lower,
        upper=upper,
        theta={"a": 1.0, "b": -0.5},
        permission=GenesisPermission(theta=2.0),
        forge=forge,
        score_fn=score_fn,
        executor_stages=[lambda eq: eq, lambda eq: eq],
        memory=None,
    )
    centroid = GeoM.centroid(final_states)
    radius = GeoM.bounding_radius(final_states, center=centroid)
    print(f"[x_engine]    admitted={pipeline_result.admitted}, "
          f"final_theta={pipeline_result.equation.theta if pipeline_result.equation else None}")
    print(f"[geom]        room-population centroid={np.round(centroid, 3)}, "
          f"bounding_radius={radius:.3f}")

    # --- 10. Patterns: remember the winning configuration ---
    memory = PatternMemory()
    winning_pattern = Pattern(
        intent="design-refinement",
        context={"theta": pipeline_result.equation.theta if pipeline_result.equation else {}},
        payload=centroid.tolist(),
        phi=1.0 - avg_safety,
        cci=0.0,
        margin=avg_safety,
    )
    memory.store(winning_pattern)
    stored = memory.all_patterns("design-refinement")
    print(f"[patterns]    stored pattern {winning_pattern.id}, "
          f"{len(stored)} pattern(s) now on file for intent 'design-refinement'")

    # --- Real, measured cost of everything above ---
    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  modules touched      : kernel, hypervisor, permission, "
          "reality_generator, space, room, agent, world, universe, runtime, "
          "validation, equation_forge, x_engine, patterns (14 modules)")
    print("  This is a classical simulation/optimization pipeline: numpy "
          "arrays evaluated by ordinary Python on this CPU. It does not "
          "create a separate physical reality and does not bypass "
          "computing cost.")


if __name__ == "__main__":
    main()
