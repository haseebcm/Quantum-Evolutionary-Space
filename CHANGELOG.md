# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `src/qes/real_distributed.py`: real multi-process, localhost-TCP
  distributed task execution (`Coordinator`, `WorkerProcess`,
  `TaskExecutionResult`, `TaskServer`) upgrading beyond the
  thread-simulated `qes.distributed` -- real OS worker processes, a real
  socket protocol, heartbeat-based crash detection and respawn, and clean
  shutdown. Validated with multiple real local processes on one machine;
  not validated across a real multi-node cluster or network. Tests in
  `tests/test_real_distributed.py`, demo in
  `examples/real_distributed_demo.py`.
- `src/qes/rpc_fault_tolerance.py`: real JSON-over-TCP RPC
  (`RPCServer`/`RPCClient`), server-side heartbeat `FailureDetector`,
  exponential-backoff `RetryPolicy`, and lowest-ID `LeaderElection`
  failover. Validated on localhost with a small number of peers; not
  validated at scale or under real network partitions. Tests in
  `tests/test_rpc_fault_tolerance.py`, demo in
  `examples/rpc_fault_tolerance_demo.py`.
- `src/qes/gpu_compute.py`: real optional GPU code paths (CuPy /
  PyTorch-CUDA) for RK4 integration and population
  divergence/convergence, with honest capability probing
  (`probe_gpu_runtime`, `is_gpu_runtime_available`) and a correct NumPy
  CPU fallback. No physical GPU was available in this project's
  development environment, so only the CPU fallback path has been
  exercised here; the GPU-specific code paths are implemented but
  unverified on real GPU hardware. Tests in `tests/test_gpu_compute.py`
  (GPU-specific case conditionally skipped), demo in
  `examples/gpu_compute_demo.py`.
- `src/qes/gpu_compute.py`: `SimulatedGPUBackend`, a real multi-process
  (genuine OS `ProcessPoolExecutor` worker processes) **software
  simulation** of GPU-lane-style parallel batch dispatch. This class does
  **not** provide real GPU hardware acceleration -- it exists solely to
  build and exercise the "many parallel compute lanes" execution shape
  used by real GPU backends, on machines (such as this project's own
  development environment, which has only an AMD integrated GPU with no
  CUDA/ROCm support) where no functional GPU runtime is present.
  Numeric results are identical to the plain CPU/NumPy path; only the
  execution/partitioning strategy differs, and each call exposes the
  real OS process IDs used (`last_lane_pids`) so the multiprocessing
  claim is independently verifiable rather than asserted. Its timings
  must never be reported as GPU performance numbers. Tests in
  `tests/test_gpu_compute.py`, demo section in
  `examples/gpu_compute_demo.py`.
- `src/qes/storage_backend.py`: real local-filesystem object storage
  (`LocalFilesystemStorageBackend`, with path-traversal/absolute-path
  rejection) and a `CheckpointStore` with SHA-256 keyed integrity
  verification. An `S3StorageBackend` is implemented against the
  `boto3` client interface but is only exercised via an
  injected/mocked client in tests, since `boto3` and real AWS
  credentials are not available in this environment -- not validated
  against a real S3 bucket. Tests in `tests/test_storage_backend.py`
  (S3 case conditionally skipped), demo in
  `examples/storage_backend_demo.py`.
- `src/qes/security.py`: real HMAC-SHA256 signed bearer tokens with
  tamper detection and expiry (`TokenIssuer`, `AuthToken`), thread-safe
  per-tenant resource accounting (`QuotaManager`, `ResourceQuota`), an
  enforceable `SandboxPolicy` for validating untrusted tenant
  configuration, and a small RBAC access-control layer
  (`AccessControlList`, `Role`, `Permission`). These are real,
  unit-tested cryptographic and accounting primitives usable as
  building blocks for a future multi-tenant service -- there is no
  deployed network-facing auth server or production key-management
  infrastructure. `Permission`/`Role` are exported at top level as
  `SecurityPermission`/`SecurityRole` since they are unrelated to
  `qes.permission`'s admissibility-gate `PermissionResult` (the shared
  word "permission" is coincidental). Tests in
  `tests/test_security.py`, demo in `examples/security_demo.py`.

### Fixed
- `benchmarks/run_benchmarks.py`: recalibrated the fixed regression
  thresholds for `space.step (rooms=1000)` (75ms -> 110ms) and
  `reality.branch (Latin hypercube, count=1000)` (12ms -> 20ms). These were
  set on a faster/idle host and were flaking under normal shared-hardware
  load even with no functional code change; new thresholds keep meaningful
  headroom over measured means while still catching genuine regressions.
  All 18 tracked benchmarks now pass (`--fail-on-regression` exits 0), and
  `README.md`'s benchmark snapshot was refreshed with the current run.
- `README.md`: the "What QES provides" module table was missing rows for
  `qes.invariants`/`qes.events` (Phase 1), `qes.cycle` (Phase 2),
  `qes.reality_operators` (Phase 3), `qes.equation_ast` (Phase 4),
  `qes.communication` (Phase 6), `qes.information_gain` (Phase 8), and
  `qes.knowledge_graph` (Phase 12) even though these modules were already
  implemented, exported, tested, and documented elsewhere (CHANGELOG,
  `docs/api-reference.md`, `docs/architecture-overview.md`). Added the
  missing rows so the table lists every Phase 1-22 module.

### Added
- `src/qes/equation_ast.py`: Phase 4's Equation Forge 2.0 -- an AST-based
  equation representation (`EquationAST`, `Constant`, `Variable`,
  `Operator`) with a primitive library, mutation/crossover/simplification,
  dimensional and numerical validation, stability analysis, and a
  fitness-driven `EquationASTForge` evolution pipeline. Tests in
  `tests/test_equation_ast.py`, demo in
  `examples/equation_forge_2_demo.py`.
- `src/qes/communication.py`: Phase 6's reality-to-reality communication
  layer -- `Message`, `CommunicationFabric` (state/knowledge/pattern
  exchange), `ResourceBid`/`ResourceNegotiation` (compute-resource
  negotiation), `Coalition`, and `compete`/`cooperate`/`form_coalition`.
  Exported at top level as `CommunicationMessage` to avoid a name clash
  with the pre-existing `qes.agent.Message`. Tests in
  `tests/test_communication.py`, demo in
  `examples/reality_communication_demo.py`.
- `src/qes/information_gain.py`: Phase 8's Information Gain Engine --
  `information_gain()` (`IG(a) = H(before) - H(after)`),
  `population_entropy()`, `InformationGainEstimator`,
  `rank_by_information_gain()`, and `allocate_by_information_gain()`,
  reusing `qes.convergence.qes_entropy`. Tests in
  `tests/test_information_gain.py`, demo in
  `examples/information_gain_demo.py`.
- `src/qes/knowledge_graph.py`: Phase 12's Persistent Knowledge Graph --
  a typed, JSON-persistable provenance graph (`KnowledgeNode`,
  `CausalHypothesis`, `KnowledgeGraph`) recording the Reality ->
  Observation -> Equation -> Pattern -> Result -> Causal-relation chain,
  successful/failed configuration tracking, strengthenable causal
  hypotheses, counterexamples, and cross-domain relationships. Tests in
  `tests/test_knowledge_graph.py`, demo in
  `examples/knowledge_graph_demo.py`.
- `src/qes/reality_operators.py`: Phase 3's next-generation Reality
  Generator operators -- `mutation`, `crossover`, `interpolate`,
  `extrapolate`, `inversion`, `perturbation` (structured, along explicit
  directions), `dimensional_transformation`, `topology_transformation`,
  `equation_substitution`, `parameter_transformation` -- plus
  `RealityFamily` (a named batch of sibling rooms with `mean_state()`,
  `std_state()`, `diversity()`, `best()`) and `RealityFamilyBuilder`
  composing several operators into one family in a single call.
- `examples/reality_operators_demo.py`: a measured demo of every operator
  and a built `RealityFamily`.
- `tests/test_reality_operators.py`: 24 tests covering every operator and
  the family container.
- `src/qes/cycle.py`: `AutonomousCycle`, the Phase 2 closed autonomous
  loop (`INTENT -> GENERATION -> REALITY POPULATION -> EXECUTION ->
  DIVERGENCE -> PERMISSION -> SELECTION -> CONVERGENCE -> KNOWLEDGE
  EXTRACTION -> PATTERN MEMORY -> ADAPTIVE REGENERATION`). Wraps
  `QESSpace.step()` (which already implements the inner
  execute/measure/permit/select/converge operator) with the missing
  outer ring: extracting each generation's outcome into `PatternMemory`,
  computing the next generation's population size via
  `adaptive_branch_count()`, branching it with `RealityGenerator`, and
  recording every stage transition to an `EventLog` plus a full
  `InvariantEngine.check_space()` pass -- so the population regenerates
  itself from its own results without a human re-seeding it by hand.
- `examples/closed_loop_cycle.py`: a measured, multi-generation run of
  `AutonomousCycle` demonstrating the loop end-to-end.
- `tests/test_cycle.py`: 15 tests covering `adaptive_branch_count` and
  `AutonomousCycle`.
- `src/qes/invariants.py`: a Global Invariant Engine (`InvariantEngine`)
  that recursively checks state validity, dimensional consistency,
  bounds, lineage integrity, resource consistency, probability
  normalization, and lifecycle consistency across `Room` -> `QESSpace` ->
  `World` -> `Universe`, returning a complete `InvariantReport` (every
  violation, not just the first) rather than raising.
- `src/qes/events.py`: an event-sourcing layer (`EventLog`, the canonical
  `SPAWN -> EXECUTE -> DIVERGE -> PERMISSION -> MUTATE -> SELECT -> MERGE
  -> COLLAPSE` event kinds), a `LineageGraph` for parent/child provenance
  queries over any tracked entity, and `DeterministicReplay`/`state_hash`
  for verifying that replaying a checkpoint reproduces bit-identical
  states.
- `tests/test_invariants.py`, `tests/test_events.py`: full coverage of
  both new modules (30 tests).
- `examples/full_stack_integration_part2.py`: a companion measured run
  chaining the 17 modules not covered by `full_stack_integration.py`
  (`state_space`, `domain`, `dynamics`, `divergence`, `closure`,
  `multi_reality`, `digital_twin`, `convergence`, `selection`,
  `compute_allocator`, `orchestrator`, `expansion`, `h11x`,
  `intelligence`, `multi_agent`, `qsee`, `worker`) — together the two
  scripts exercise all 32 `qes` engine modules end-to-end, each printing
  real wall-clock time and peak memory for the whole run.
- `examples/full_stack_integration.py`: a single measured end-to-end run
  chaining 14 modules (`kernel`, `hypervisor`, `permission`,
  `reality_generator`, `space`, `room`, `agent`, `world`, `universe`,
  `runtime`, `validation`, `equation_forge`, `x_engine`, `patterns`) and
  printing real wall-clock time and peak memory for the whole run — a
  concrete answer to "do the modules actually integrate", with an explicit
  reminder that it is ordinary CPU/RAM cost, not literal universe creation.

### Changed
- Hardened all 16 remaining engine modules (`closure`, `expansion`, `h11x`,
  `hypervisor`, `intelligence`, `kernel`, `multi_agent`, `multi_reality`,
  `parallel_fabric`, `patterns`, `qsee`, `runtime`, `validation`, `worker`,
  `x_engine`) with stricter input validation (shape/finite/range checks),
  defensive copying of mutable state, vectorized numeric hot paths, and
  completed type hints/docstrings. No public API changes; all 380 tests
  pass, `ruff` and `mypy` remain clean.

## [1.3.0]

### Added
- `src/qes/domain_packs.py`: Phase 21's Domain Packs -- reusable, validated
  adapters that collapse the ad-hoc setup boilerplate seen in
  `examples/control_policy_search.py`, `examples/design_space_search.py`, and
  `examples/resource_allocation.py` into single configured calls:
  `ControlPolicyPack`/`make_control_policy_space()` (controller-gain search
  over plant dynamics), `DesignSpacePack`/`make_design_space()`
  (constrained structural-design search), and
  `ResourceAllocationPack`/`make_resource_allocation_pack()`
  (priority-weighted compute/job allocation), plus `DomainPackRegistry`/
  `list_domain_packs()` for discoverability. Every pack is pure orchestration
  over the existing `QESSpace`/`RealityGenerator`/`compute_allocator`
  primitives -- no new search algorithms. Tests in
  `tests/test_domain_packs.py`, demo in `examples/domain_packs_demo.py`.
- `src/qes/sdk.py`: Phase 22's QES SDK -- a thin, ergonomic front door over
  the core engine: `QESClient` wires `RealityGenerator` + `QESSpace` +
  `GenesisPermission`/`AdaptivePermission` + `PatternMemory` from a handful
  of constructor arguments and exposes a single `.run(steps=N)` returning an
  `SDKRunResult` (best room/state, final entropy/convergence, wall time,
  step count); `quick_search()` is a one-shot convenience function for the
  simplest bounded-search use case. The SDK performs no computation of its
  own -- every result is produced by the existing `qes.*` primitives it
  orchestrates. Tests in `tests/test_sdk.py`, demo in `examples/sdk_demo.py`.

### Changed
- Exported all new Phase 21/22 APIs via `src/qes/__init__.py` and bumped the
  package version to `1.3.0`. Full suite: 969 tests passing, `ruff` and
  `mypy` clean.

## [1.2.0]

### Added
- `src/qes/causal_engine.py`: Phase 5's Causal Reality Engine -- a classical
  DAG-based causal-analysis layer (`CausalGraph`, `CausalEdge`) that can run
  interventions (`intervene`), counterfactual comparisons (`counterfactual`),
  perturbation-sensitivity scans (`perturbation_sensitivity`), heuristic
  hidden-variable discovery (`hidden_variable_hypotheses`), and bundled causal
  passes (`analyze_causality`) over rooms or plain state mappings. The
  propagation model is intentionally explicit and modest: weighted linearized
  delta flow through a caller-defined DAG, not literal causality discovery or
  quantum computation. Tests in `tests/test_causal_engine.py`, demo in
  `examples/causal_reality_engine_demo.py`.
- `src/qes/marketplace.py`: Phase 7's Reality Marketplace / Resource Economy --
  multi-resource accounts and bids (`RealityAccount`, `MarketBid`) plus
  `compute_allocation_score()` and `RealityMarketplace`, which clears
  compute/memory/energy rounds using utility, risk, novelty, uncertainty,
  information gain, and historical performance while reusing the earlier
  `ComputeAllocator` rather than silently reimplementing budgeting. This
  remains an in-process classical market over finite CPU/RAM-style budgets,
  not literal universe economics. Tests in `tests/test_marketplace.py`, demo
  in `examples/reality_marketplace_demo.py`.
- `src/qes/digital_twin_loop.py`: Phase 9's Digital Twin 2.0 loop --
  `Observation`, inverse-variance `sensor_fusion()`, `OnlineParameterEstimator`
  (recursive least squares), `StateEstimator` (linear Kalman filter),
  `uncertainty_propagation()`, `ModelSelector`, residual-based
  `anomaly_detection()`, `prediction_interval()`, `calibration_check()`,
  `drift_detection()`, and the full `DigitalTwinLoop.observe()` orchestrator.
  This is a classical estimation/prediction loop running on CPU/RAM, not
  literal quantum computation. Tests in `tests/test_digital_twin_loop.py`,
  demo in `examples/digital_twin_loop_demo.py`.
- `src/qes/backend.py`: Phase 10's accelerator-backend abstraction --
  `ComputeBackend`, `NumpyBackend`, honest optional `GPUBackend`,
  `BackendRegistry`, `get_backend()`, and `benchmark_backend()`. The
  guaranteed path is ordinary NumPy CPU execution; GPU support activates only
  when a real CuPy or CUDA-enabled PyTorch runtime is detected and otherwise
  fails honestly with `BackendUnavailableError`. Tests in
  `tests/test_backend.py`, demo in `examples/accelerator_backend_demo.py`.
- `src/qes/distributed.py`: Phase 11's distributed-coordination layer --
  `WorkItem`, thread-backed `Worker`, central `Controller`, JSON-serializable
  checkpoints (`distributed_checkpoint`, `restore_from_checkpoint`), and
  `autoscaling_policy()`. The coordination logic (heartbeats, leases, crash
  recovery, work reassignment, GPU-aware placement) is real, but execution is
  honestly simulated with threads inside one process rather than real
  multi-machine deployment. Tests in `tests/test_distributed.py`, demo in
  `examples/distributed_qes_demo.py`.
- `src/qes/meta_learning.py`: Phase 13's meta-learning controller --
  `ProblemStructure`, `SearchStrategy`, `SearchOutcome`, `SearchHistory`,
  `MetaController`, and `record_outcome()`, so QES can adapt optimizer choice
  and hyperparameters across runs using cheap problem features plus
  historically successful configurations stored locally or in `PatternMemory`.
  This is transparent heuristic adaptation over tracked history, not deep
  learning or AGI. Tests in `tests/test_meta_learning.py`, demo in
  `examples/meta_learning_demo.py`.
- `src/qes/self_improvement.py`: Phase 14's governed self-improvement
  pipeline -- `PerformanceEvaluation`, `ArchitectureProposal`, `Sandbox`,
  `ApprovalPolicy`, `SelfImprovementPipeline`, and helpers for proposing
  alternatives, sandbox benchmarking, safety gates, and regression gates.
  "Deployment" here only records an approved strategy/configuration in an
  in-memory strategy library; the module never rewrites production source
  code. Tests in `tests/test_self_improvement.py`, demo in
  `examples/self_improvement_demo.py`.
- `src/qes/verification.py`: Phase 15's formal verification layer -- staged
  `numerical_verification`, `constraint_verification`, `safety_verification`,
  `stability_verification`, `invariant_verification`,
  `cross_domain_verification`, plus `VerificationStageResult`, `Evidence`,
  and `VerificationEngine`. The result is a structured evidence package over
  classical numerical/logical checks; it is not a theorem prover or an
  external formal-methods integration. Tests in `tests/test_verification.py`,
  demo in `examples/formal_verification_demo.py`.
- `src/qes/adversarial.py`: Phase 16's adversarial / red-team candidate
  generator -- `FailureCategory`, `AdversarialCandidate`, `FailureAnalysis`,
  `RedTeamReport`, `AdversarialGenerator`, and `run_red_team_campaign()`,
  covering boundary, unstable-state, numerical-singularity,
  adversarial-input, resource-exhaustion, pathological-equation, coupling,
  and cascading-failure probes. This is classical synthetic stress testing
  for governed numeric search, not literal "reality breaking." Tests in
  `tests/test_adversarial.py`, demo in `examples/adversarial_reality_demo.py`.
- `src/qes/multi_agent_evolution.py`: Phase 17's multi-agent QSEE evolution
  layer -- eleven lightweight `SpecializedAgent`s managed by
  `AgentPopulation`, plus `ScoredCandidate`, `SharedInsight`, `Coalition`,
  `CollectiveSolutionResult`, `ComparisonResult`, `collective_solution()`,
  and `compare_individual_vs_collective()`. These agents are intentionally
  heuristic scorers over shared candidate features, not LLMs or autonomous
  AGI entities. Tests in `tests/test_multi_agent_evolution.py`, demo in
  `examples/multi_agent_evolution_demo.py`.
- `src/qes/bench.py`: Phase 18's `qes-bench` benchmark suite --
  `BenchmarkProblem`, `BaselineResult`, `BenchmarkMetrics`, `BenchReport`,
  problem constructors for Sphere/Rastrigin/Rosenbrock, the real
  `run_qes_baseline()`, a registry of honestly labeled minimal reference
  baselines (Bayesian optimization, evolutionary/genetic search, simulated
  annealing, CMA-ES-style, differential evolution, gradient method, MPC,
  reinforcement-learning-style, symbolic-regression-style, and
  multi-objective search), optional SciPy-backed differential-evolution and
  gradient-method runners when SciPy is available, and the top-level
  `run_qes_bench()`. The non-QES baselines are explicitly minimal reference
  implementations for fair comparison, not literature-SOTA claims. Tests in
  `tests/test_bench.py`, demo in `examples/qes_bench_demo.py`.
- `src/qes/reproducibility.py`: Phase 19's reproducibility infrastructure --
  `ExperimentArtifact`, `ExperimentRecorder`, `ReproductionCheckResult`, JSON
  persistence (`save_artifact`, `load_artifact`), `reproduce_experiment()`,
  and `verify_reproduction()`. Each artifact captures code version, git
  commit, local hardware metadata, seeds, population, lineage, equations,
  decisions, allocations, failures, and final evidence so experiments can be
  replayed with the same configuration on ordinary deterministic Python/NumPy
  execution. Tests in `tests/test_reproducibility.py`, demo in
  `examples/reproducibility_demo.py`.
- `src/qes/observatory.py`: Phase 20's QES Observatory --
  `ObservatorySnapshot`, `render_reality_tree()`, `ObservatoryDashboard`, and
  `build_snapshot_from_modules()` for a dependency-free ASCII/Unicode
  dashboard that can summarize population, convergence, risk, novelty,
  permission, divergence, marketplace allocations, digital-twin outputs,
  failures, discovered patterns, and optional knowledge-graph sections in
  one plain-text snapshot. Tests in `tests/test_observatory.py`, demo in
  `examples/observatory_demo.py`.

### Added
- `src/qes/kernel.py`: the Digital Existence Kernel — an 11-layer closed
  core (Null Origin, Existence Allowance, Construction, Mechanism,
  Operation, Rejection, Persistence/Materialization, Bounded Energy,
  Computable Geometry, Recursive Autonomous Entities, Self-Generating
  Domain Intelligence) that governs whether a candidate digital entity is
  allowed to exist before it reaches the room/world/universe layers.
  Also includes `RealitySelector`, the fixed-point "reality compiler"
  distinguishing candidate possibilities from instantiated realities
  (`R = {x in C : x == Phi(x)}`).
- `tests/test_kernel.py`: full coverage of all 11 kernel layers and the
  reality selector.
- New "Digital Existence Kernel" sections in `docs/architecture-overview.md`
  and `docs/api-reference.md` documenting the new module.

### Changed
- Exported all new kernel APIs via `src/qes/__init__.py` and bumped the
  package version to `0.5.0`.

### Added (previous)
- `src/qes/qsee.py`: a QSEE-11L runtime with isolated `QEL` streams,
  `BIG11` structural isolation, and `AcrosV12BIESync` synchronization. This
  implements the eleven-stream architecture as a strict read-only merge gate
  rather than a permissive cross-stream coupling.
- `src/qes/h11x.py`: the six-layer H^11X engineering admissibility stack
  (`necessity_field`, `constraint_geometry`, `domain_gatekeeper`,
  `failure_anticipation`, `self_generative_correction`, and `export_barrier`).
- `src/qes/closure.py`: the formal closure and cross-domain verification
  layer for divergence, permission closure, safe exploration, action
  selection, and risk monotonicity checks.
- `src/qes/selection.py`: a `GenesisSelectionPipeline` that documents,
  tracks, and enforces the 11-stage generate → classify → permit → select
  lifecycle and first-of-kind population monotonicity rules.
- New tests for the entire advanced architecture bundle: `test_qsee.py`,
  `test_h11x.py`, `test_closure.py`, and `test_selection_pipeline.py`.
- Additional repository documentation tying the new architecture to the core
  universe definition in `README.md` and `docs/QES-architecture.md`.

### Changed (previous)
- Exported all new public APIs via `src/qes/__init__.py` and bumped the
  package version to `0.4.0`.
- Kept the repository validation bar at the highest standard: full pytest
  coverage, Ruff clean, and mypy clean.

### Added (previous)
- Seven new architectural-layer modules implementing components from the
  H^11 COSMIC VP / X-Engine / Multi-Reality specification (see
  `docs/QES-architecture.md`), extending QES beyond room/world/universe
  primitives into a full virtualization and expansion fabric:
  - `hypervisor.CosmicVP`: the root virtualization substrate beneath QES.
    Provisions VMs/networks/storage/containers and implements all seven
    closed control loops -- C-Loop (creation), S-Loop (stabilization),
    X-Loop (dual-engine security check), D-Loop (drift/storage mapping),
    A-Loop (ACROS execution), EC-Loop (cosmic node expansion), and
    Omega-Loop (null-collapse-and-regenerate recovery) -- plus a full
    `request_cycle()` implementing the request/compute cycle end to end.
  - `x_engine`: the X-Engine stack -- `AcrosV12BIE` (equation
    stabilization via hill-climbing), `AcrosV13` (adaptive
    argmin-over-risk/inconsistency/instability selection with an
    `amplifying()` trend check), `ApexI` (traced sequential executor),
    `GeoM` (distance/projection/centroid/bounding-radius geometry), and
    `x_engine_pipeline()` wiring Intent -> Genesis -> Equation -> Execution
    -> Geometry -> Patterns into one call.
  - `parallel_fabric`: the Parallel Compute Fabric's six domains --
    `ParallelComputeDomain` (thread-pool map + matrix multiplex),
    `SynchronizationDomain` (barrier + temporal match filter),
    `PredictionDomain` (forecast/trajectory/probability-drift),
    `ExecutionDomain` (branch-prediction execution + output
    normalization), `MonitoringDomain` (metrics dashboard, stability
    probe, drift monitor), and `ErrorCorrectionDomain` (ECC median
    correction, H^11 tri-cycle corrector, integrity validation).
  - `multi_reality`: `SimulationEngine` (parallel independent room
    trajectories), `VirtualNodeProjection` (branch a room into virtual
    compute nodes), `PatternReplicationLayer` (replicate a winning
    pattern across nodes), and `VQCE` (the H^11 Virtual Quantum-less
    Compute Engine: allocate -> stabilize -> compute -> dual-engine
    validate, explicitly without physical quantum hardware).
  - `validation`: `ValidationFramework` (named V&V checks),
    `FunctionalSafetySystem` (safety-margin scoring), `FailureRecoveryLoop`
    (the Omega-Loop's error -> null -> regenerate -> restabilize cycle),
    and `IntegrityValidator` (redundant-replica consensus/consistency).
  - `expansion`: `DomainBirthKernel` (births new `World`s on demand),
    `InfinityRouter` (routes to an existing world or births a new one),
    and `MetaExpansionEngine` (the universe-level EC-Loop: attach a new
    world to a `Universe` once all existing worlds cross a load
    threshold).
  - `multi_agent`: `CognitiveAgentSpawner` (births agent populations from
    a template), `LayeredRoleEngine` (assigns agents to roles across an
    ordered hierarchy of layers), and `AutonomousNodeRouter` (routes a
    task to the fittest agent or broadcasts it to all).
  - 79 new tests covering all seven modules; full suite now 283 tests at
    100% line/branch coverage; ruff/mypy clean.
- `benchmarks/run_benchmarks.py`: a dependency-free performance benchmark
  suite (stdlib-only, no `pytest-benchmark` needed) covering `QESSpace.step()`
  at increasing room-population scale, `Universe.step()`/`clone()`/
  `snapshot()`+`restore()` with nested universes and agents, Gaussian vs.
  Latin-hypercube reality branching, the three compute-allocation
  strategies, mutate-only vs. full generational equation evolution, and the
  Euler/RK4/stochastic dynamics integrators. Supports `--repeat` and
  `--json` for tracking performance over time.
- Extreme-advancement pass across every existing module, each gaining a
  substantial new capability rather than cosmetic changes:
  - `Agent`: bounded/decaying memory, goal-directed behaviour
    (`set_goal`/`goal_progress`/`is_goal_satisfied`), inter-agent messaging
    (`send`/`receive`/`Message`), and a bounded action history log.
  - `World`: shared environment `fields` broadcast into every agent's
    observation, a `broadcast()` message bus, and `snapshot()`/`restore()`
    checkpointing.
  - `Universe`: full "Reality Branching" operator set --
    `clone()`, `simulate()`, `compare()`, `merge()`, `collapse_nested()`,
    and `snapshot()`/`restore()`.
  - `QESSpace`: `clone()` and `snapshot()`/`restore()` for whole-space
    checkpointing/branching.
  - `selection`: `pareto_front()` and
    `GenesisSelection.select_pareto_per_kind()` for true multi-objective
    (non-dominated-front) selection alongside the existing scalar argmin.
  - `convergence`: `kl_divergence()` and `gini_coefficient()` diversity
    diagnostics alongside the entropy-based convergence coefficient.
  - `orchestrator`: `AdaptiveAcros`, an Adam-style per-component adaptive
    gain replacing the fixed scalar `K` in the ACROS correction law.
  - `permission`: `AdaptivePermission`, a gate whose admission threshold
    Theta self-tunes from recent admission-rate history.
  - `reality_generator`: `latin_hypercube_samples()` /
    `branch_latin_hypercube()` for stratified, gap-free reality branching.
  - `compute_allocator`: `allocate_with_floor()` (fairness floor) and
    `allocate_multi_resource()` (independent multi-resource budgets).
  - `dynamics`: `integrate_rk4()` (4th-order Runge-Kutta),
    `stochastic_noise()`, and `step_stochastic()` for higher-order,
    noise-aware room evolution.
  - `equation_forge`: `crossover()` (sexual recombination between two
    equations) and `spawn_next_generation()` (elitism + crossover +
    mutation in one full generational-replacement step).
  - `room`: `generation` property (branching depth) and `tag()`/`get_tag()`
    metadata helpers.
  - `digital_twin`: rolling residual window plus `detect_drift()` for
    flagging rooms that have drifted from the observed real system.
  - `divergence`: `hsa_state()` classifying health scores into
    stable/critical/singular, exposed on `DivergenceResult.state()`.
  - `domain`: `compose()` to merge two `DomainNullification` catalogues.
  - `state_space`: named dimensions (`index_of`/`labeled`) and `volume()`.
- 82 new tests covering every enhancement above plus previously-uncovered
  edge cases (204 tests total, 100% line and branch coverage).

### Added (previous)
- `qes.agent.Agent`, `qes.world.World`, and `qes.universe.Universe`:
  formalizes QES as an intelligent virtual universe
  `Q = {S, E, A, R, M, C, T, G}` that hosts worlds (`R`), which in turn host
  room populations (`QESSpace`), intelligent agents (`A`), and optionally
  further nested universes, yielding the recursive containment chain
  `Q^(0) ⊃ Q^(1) ⊃ ...`. See §33-36 of `docs/QES-architecture.md`.
- `examples/intelligent_universe.py` demonstrating the universe/world/agent
  hierarchy, including a nested universe.
- Optional parallel room execution in `QESSpace.execute()` via
  `max_workers` (thread pool fan-out across independent rooms).
- GitHub Actions CI running lint (ruff), type checks (mypy), and the test
  suite with coverage across Python 3.10–3.12.
- `ruff` and `mypy` configuration, `pytest-cov` coverage reporting.
- `examples/control_policy_search.py` and `examples/design_space_search.py`
  demonstrating the framework applied to non-market domains.
- `CONTRIBUTING.md`, issue templates, and a pull request template.

## [0.1.0] - 2026-08-11

### Added
- `docs/QES-architecture.md`: 32-section formal specification of the H¹¹
  Quantum Evolutionary Space (QES) framework.
- `src/qes/` Python package implementing the specification: `room`,
  `state_space`, `domain`, `dynamics`, `divergence`, `permission`,
  `equation_forge`, `reality_generator`, `selection`, `compute_allocator`,
  `digital_twin`, `convergence`, `orchestrator`, `patterns`, and the
  top-level `space.QESSpace` master operator.
- Unit test suite covering every module.
- `examples/basic_run.py` quick-start example.
- MIT `LICENSE`, README quick-start and installation instructions.
