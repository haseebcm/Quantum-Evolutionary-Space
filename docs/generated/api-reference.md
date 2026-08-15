# QES API Reference

QES exposes a small but expressive public surface centered on bounded candidate realities, governed search loops, and higher-level runtime orchestration.

## Public entry points

```python
from qes import (
    Room,
    QESSpace,
    GenesisPermission,
    RealityGenerator,
    GenesisSelectionPipeline,
    ProductionRuntime,
    Universe,
    World,
)
```

## Core concepts

- `Room`: the canonical candidate reality.
- `QESSpace`: the master coordinator for generation, execution, permission checking, and convergence.
- `GenesisPermission`: the admissibility gate that decides whether a candidate remains viable.
- `Universe` / `World`: recursive runtime containers for higher-level orchestration.
- `ProductionRuntime`: deployment-like runtime for job scheduling and persistence.

## API by domain

### Core search and state

- `Room` — Canonical bounded candidate reality. Carries state vector, bounds, activation, lineage, metadata, and dynamic weight used during permission and selection.
- `MCCStateSpace` — State-space envelope defining structured bounds and admissible domain geometry for a room.
- `DomainNullification` — Aggregates domain metrics into one active signal used by divergence and evaluation.
- `QESSpace` — Master search engine that executes generation, divergence, permission, selection, and convergence over a room population.
- `RealityGenerator` — Creates and mutates room populations from a seed room using stochastic or structured branching.
- `GenesisPermission` — Admissibility kernel enforcing safe ranges, violation energy, and permission margins.
- `AdaptivePermission` — Genesis permission gate whose admission threshold Theta tightens or relaxes based on the observed admission rate.
- `PermissionResult` — Bundled result of a permission evaluation: violation energy, CCI, margin, hard/soft permission, and admission flag.

### Selection and optimization

- `GenesisSelection` — Selection primitive for ranking candidate rooms by objective, admissibility, and survival criteria.
- `GenesisSelectionPipeline` — Layered pipeline that stages generation, classification, permission, and final survival selection.
- `pareto_front` — Computes Pareto-efficient candidates among a room population.
- `ComputeAllocator` — Distributes a compute budget across rooms by weighted priority, fairness floor, and multi-resource constraints.
- `DigitalTwin` — Compares simulated experience against observed reality to update evidence weights.
- `KindTrace` — Per-kind trace record: signature, first-seen layer, survivor candidate, score, elimination layer, population history.
- `PipelineResult` — Genesis-Selection pipeline's final outcome: per-kind traces plus the monotone-shrinking population history.

### Adaptive search intelligence

- `AdaptiveGradientSearch` — Governed step_fn combining central-difference gradient estimation with an Adam-style per-dimension adaptive update, Rechenberg 1/5-rule self-adaptive stochastic search, and stagnation-triggered restarts. QES's default intelligent local/global search strategy.
- `AdaptiveSearchConfig` — Configurable hyperparameters for AdaptiveGradientSearch: step-size bounds, Adam moment coefficients, gradient probability, and stagnation tuning.
- `OptimizationResult` — Outcome of optimize(): best point found, best value, iteration count, evaluation count, and surviving room count.
- `optimize` — End-to-end convenience entry point wiring RealityGenerator branching, a permission-gated QESSpace, and AdaptiveGradientSearch together for any bounded, continuous objective.

### Orchestration and lifecycle

- `Acros` — Self-stabilizing orchestration law applying the correction gradient xdot = f(x, t) - chi*K*grad[Phi + mu*CCI] to steer a room toward admissibility.
- `AdaptiveAcros` — Acros with an Adam-style per-component adaptive gain replacing the fixed gain K, for ill-conditioned objectives.
- `RoomLifecycle` — Validates and applies allowed room lifecycle state transitions across the seven defined lifecycle states.
- `Equation` — Single equation hypothesis carrying parameters, fitness, lineage parent, generation depth, and an optional callable.
- `EquationForge` — SGEE population genetics engine: seeds equations and recursively mutates, crosses over, and selects populations of equation hypotheses.

### Divergence and governance

- `DSA` — Divergence and stability tracker used to evaluate how far a room drifts from its target domain.
- `DivergenceResult` — Structured divergence report containing drift, hazard, and stability data.
- `divergence` — Finite-distance divergence measure used to score admissibility and collapse risk.
- `divergence_gradient` — Gradient of divergence used for local correction and risk-aware exploration.
- `collapse_proximity` — Risk metric combining collapse pressure and proximity to failure states.
- `hsa_state` — Classifies a health score S_i(t) into stable/critical/singular based on critical-band tolerance.
- `PermissionClosure` — Filters candidate actions to admissible ones under constraints and closure rules.
- `PermissionClosureResult` — Bundled output of PermissionClosure.evaluate(): the admissible action set and a safety flag.
- `action_selection` — Selects the action minimizing J(x,u) + lambda*S(F(x,u,xi)) from the admissible action set A(x).
- `safe_exploration_closure` — Returns True iff a policy-search operator maps every admissible policy back to an admissible policy.
- `CrossDomainVerifier` — Checks reconstruction, state closure, and constraint closure across domains.
- `DomainVerificationResult` — Verification result: reconstruction error plus state-closure and constraint-closure booleans.

### Advanced architecture

- `Universe` — Top-level recursive digital space containing worlds, agents, and nested universes.
- `World` — Runtime environment hosting rooms, agents, and nested entities.
- `Agent` — Autonomous inhabitant that perceives, decides, and acts within a world or universe.
- `Message` — Single inter-agent message carrying sender id, payload, and timestamp for inbox delivery.
- `PatternMemory` — Stores dominant and non-dominated patterns for reuse across exploration cycles.
- `CosmicVP` — Root hypervisor style substrate for virtualized resource and control loops.
- `RequestCycleResult` — Full COSMIC VP cycle outcome: request, output, stabilized/expanded/recovered flags, and provisioned resources.
- `VirtualResource` — Single provisioned virtual resource (VM, network, storage, or container) with kind, spec, and unique id.
- `QSEE11L` — Eleven-stream intelligence runtime with isolated streams and synchronized output.
- `BIG11` — Bipolar Isolation Gate enforcing structural isolation between eleven independent QEL evolution streams.
- `QEL` — One isolated QSEE evolution layer/stream: private state and characteristic vector, delta history, no cross-stream access.
- `AcrosV12BIESync` — ACROS V12-BIE synchronization gate: the unique point where eleven isolated QEL states sequentially meet.
- `SyncResult` — Synchronization gate output: the aligned per-stream states in order, plus a deterministic combined output.
- `H11X` — Six-layer engineering admissibility gate for necessity, geometry, failure anticipation, and correction.
- `H11XResult` — Full six-layer evaluation outcome: admission status, geometry, correction, export, and denied layer (if any).
- `Geometry` — Layer 2 structural geometry: stress, load paths, energy flow, temporal stability, and a severity measure.
- `constraint_geometry` — Layer 2: combines a need input and a constraint input into an emergent geometry representation.
- `necessity_field` — Layer 1: returns True iff N(x) >= 0, permitting formation to proceed.
- `domain_gatekeeper` — Layer 3: denies unless every domain check passes (a single failing domain fails the gate).
- `failure_anticipation` — Layer 4: pushes a candidate toward failure, denying it unless a recovery path exists.
- `CorrectionOutcome` — Layer 5 regeneration output: reintegrated, relaxed, and reformed failure-recovery states.
- `self_generative_correction` — Layer 5: runs the failure -> reintegration -> constraint relaxation -> reformation pipeline.
- `ExportResult` — Layer 6 output container holding only the derived artifact, never internal H^11X state.
- `export_barrier` — Layer 6: derives a result from internal state but returns only the derived output, never the internal state itself.
- `AcrosV12BIE` — Equation perfection/stabilization via hill-climbing mutations, keeping only fitness-improving candidates.
- `AcrosV13` — Adaptive logic amplification/evolution, selecting the configuration minimizing Risk + Inconsistency + Instability.
- `ApexI` — Power/core executor: pipelines callables in sequence, threading output to input, recording a per-stage trace.
- `GeoM` — H^11 geometry engine: weighted distance, orthogonal projection, centroid, and bounding-radius operations.
- `XEnginePipelineResult` — Full X-Engine pipeline output: intent, admission, equation, execution, geometry, and pattern data.
- `x_engine_pipeline` — Complete Intent -> Genesis -> Equation -> Evolution -> Execution -> Geometry -> Patterns flow in one call.
- `AutonomousNodeRouter` — Routes tasks to the best-fit agent without a central scheduler, using a fitness function.
- `CognitiveAgentSpawner` — Births a population of agents from a shared policy template, each with its own per-agent state.
- `LayeredRoleEngine` — Stratifies an agent population into a role hierarchy via round-robin role assignment across layers.
- `VQCE` — H^11 Virtual Quantum-less Compute Engine: stabilizes, computes, and validates without physical quantum hardware.
- `PatternReplicationLayer` — Replicates a winning pattern across nodes, seeding room memory with the validated configuration.
- `SimulationEngine` — Runs a single room forward under multiple candidate step functions in parallel realities.
- `VirtualNodeProjection` — Projects a room into a virtual node population, one node per branched reality.
- `ErrorCorrectionDomain` — ECC + Tri-Cycle Corrector + Integrity Validator for redundant-replica consensus and correction.
- `ExecutionDomain` — Instruction Executor + Branch Prediction Engine + Output Normalizer for branched execution.
- `MonitoringDomain` — Metrics Dashboard + Stability Probe + Drift Monitor for telemetry and health tracking.
- `ParallelComputeDomain` — MetaGPI Core + Thread Pool Generator + Matrix Multiplex Engine for parallel computation.
- `PredictionDomain` — Forecast Engine + Pattern Trajectory Mapper + Probability Drift Layer for time-series prediction.
- `SynchronizationDomain` — Thread Synchronizer + Temporal Match Filter for multi-thread coordination and temporal alignment.
- `UniverseComparison` — Telemetry-level diff between two universes: time/world/agent/equation/event deltas.
- `UniverseTelemetry` — Snapshot of Q(t): time, world count, agent count, equation count, and event count.
- `DomainBirthKernel` — Instantiates brand-new worlds/domains on demand, spawning fresh ones or cloning from parents.
- `InfinityRouter` — Routes events to worlds with capacity, birthing new worlds when all existing ones are full.
- `MetaExpansionEngine` — EC-Loop universe expansion: births new worlds when existing ones exceed a load threshold.

### Runtime and deployment

- `ProductionRuntime` — High-level runtime facade for orchestrating QES jobs with priority scheduling and state persistence.
- `RuntimeScheduler` — Executes queued jobs in priority order with optional worker threads.
- `RuntimeJob` — Runnable unit of work: name, handler, priority, metadata, status, attempts, result, and error.
- `RuntimeStore` — In-memory store for runtime metadata and job outcomes.
- `FileRuntimeStore` — JSON-backed store for durable runtime state across process boundaries.
- `SQLiteRuntimeStore` — Thread-safe SQLite-backed durable store with lock-serialized access for concurrent workers.
- `RedisRuntimeStore` — Redis-backed runtime store suitable for horizontally scaled multi-process/multi-node deployments.
- `PostgresRuntimeStore` — Postgres-backed durable runtime store with transactional consistency for multi-node worker pools.
- `DistributedRuntime` — Deployment-oriented runtime facade extending ProductionRuntime with multi-worker orchestration.
- `ApiKeyGuard` — Minimal constant-time HMAC-compared API-key access control for production runtime deployments.
- `RuntimeAccessError` — Permission error raised on API-key validation failure during runtime access.
- `RuntimeTelemetry` — Summary of queue health, completion rates, and runtime performance.

### Validation, safety, and metrics

- `ValidationFramework` — V&V framework: runs a battery of named boolean checks against a state and reports failures.
- `FunctionalSafetySystem` — Scores how far a state sits inside a safety-margin envelope: 1.0 centered, 0.0 at the boundary.
- `IntegrityValidator` — Cross-checks redundant replicas for consistency via componentwise median-consensus voting.
- `FailureRecoveryLoop` — Omega-Loop: error detection -> null baseline -> regeneration -> restabilization within the envelope.
- `convergence_coefficient` — C_Q = 1 - H_bar_Q, measuring how far room weights have converged from uniform dispersion.
- `qes_entropy` — Shannon entropy H_Q = -sum_i p_i * ln(p_i) of the room-weight distribution.
- `kl_divergence` — Kullback-Leibler divergence KL(p || q) measuring drift of current weights from a reference distribution.
- `gini_coefficient` — Gini inequality index of the weight distribution in [0, 1], sensitive to tail concentration.

## Typical usage patterns

### Search loop

```python
from qes import GenesisPermission, QESSpace, RealityGenerator, Room
import numpy as np

seed = Room(
    x=np.zeros(3),
    x_star=np.zeros(3),
    lower=-np.ones(3),
    upper=np.ones(3),
    activation=np.ones(3),
)

generator = RealityGenerator()
candidates = generator.branch(seed, count=25, scale=0.3)
space = QESSpace(permission_gate=GenesisPermission(theta=1.5), dt=1.0)
space.spawn(candidates)
telemetry = space.step()
print(telemetry)
```

### Adaptive intelligent search

```python
from qes import Room, optimize
import numpy as np

seed = Room(
    x=np.full(6, -1.0),
    x_star=np.ones(6),
    lower=-np.ones(6) * 5,
    upper=np.ones(6) * 5,
    activation=np.ones(6),
)

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

result = optimize(rosenbrock, seed, iterations=300, population=40)
print(result.best_x, result.best_value)
```

`optimize()` runs QES's adaptive gradient search (Adam-style per-dimension steps + self-adaptive stochastic search + stagnation restarts) through the full governed room/space loop, and is competitive with derivative-free external baselines like SciPy's Nelder-Mead on standard benchmarks such as the Rosenbrock function (see `examples/scipy_baseline_benchmark.py`).

### Runtime scheduling

```python
from qes import ProductionRuntime

runtime = ProductionRuntime(max_workers=4)
runtime.submit('task-a', lambda: {'ok': True}, priority=5)
for job in runtime.run_all():
    print(job.status, job.result)
```

### Recursive universe composition

```python
from qes import Universe, World

u = Universe()
w = World()
u.add_world(w)
print(u)
```

## Notes

This reference is intentionally organized by responsibility rather than by implementation file. The package is designed to be composable: room dynamics, admissibility policies, runtime orchestration, and higher-level recursive structures can be mixed without changing the core architecture.
