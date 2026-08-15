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
- `AdaptivePermission` — Genesis permission gate whose admission threshold `Theta` tightens or relaxes based on the observed admission rate.
- `PermissionResult` — Bundled result of a permission evaluation: violation energy, CCI, margin, hard/soft permission, and admission flag.

### Digital Existence Kernel

The 11-layer closed core underneath the rest of QES — governs whether a
candidate digital entity is even allowed to exist before it reaches the
higher search/permission/selection layers. See
[architecture-overview.md](architecture-overview.md#digital-existence-kernel)
for the full layer-by-layer narrative.

- `ExistenceKernel` — Layers 1-8: null origin, existence allowance, bounded construction, identity-preserving mechanism, governed operation, rejection, persistence/materialization, and bounded computational energy.
- `RejectionError` — Raised when an operation, construction step, or energy budget would produce a non-admissible state; the state never comes into existence.
- `ConstructionResult` — Result of one bounded-construction step: accumulated complexity and the applied delta.
- `OperationResult` — Result of one governed operation: the resulting state and whether identity/coherence was preserved.
- `PersistenceResult` — Result of iterating an operator toward a fixed point: converged state, iteration count, and residual.
- `ComputableGeometry` — Layer 9: computable support-set geometry for an entity, with non-overlap checks between entities.
- `RecursiveEntity` — Layer 10: recursive autonomous entity with bounded divergence per step and fixed-point reproduction detection.
- `RecursiveEntityResult` — Result of one recursive-entity step: next state, divergence magnitude, and whether it stayed within bound.
- `SelfGeneratingDomainIntelligence` — Layer 11: generates new computational domains from existing ones while preserving the core admissibility invariant.
- `DomainGenerationResult` — Result of one domain-generation step: the generated domains and whether all remained admissible.
- `RealitySelector` — Reality compiler: filters a candidate possibility space down to the fixed-point subset `R = {x in C : x == Phi(x)}` that survives as instantiated reality.
- `RealitySelectionResult` — Result of `RealitySelector.select`: the surviving realities and the rejected (non-instantiated) candidates.

### Selection and optimization

- `GenesisSelection` — Selection primitive for ranking candidate rooms by objective, admissibility, and survival criteria.
- `GenesisSelectionPipeline` — Layered pipeline that stages generation, classification, permission, and final survival selection.
- `pareto_front` — Computes Pareto-efficient candidates among a room population.
- `ComputeAllocator` — Distributes a compute budget across rooms by weighted priority, fairness floor, and multi-resource constraints.
- `DigitalTwin` — Compares simulated experience against observed reality to update evidence weights.
- `KindTrace` — Per-kind trace record: signature, first-seen layer, survivor candidate, score, elimination layer, population history.
- `PipelineResult` — Genesis-Selection pipeline's final outcome: per-kind traces plus the monotone-shrinking population history.

### Adaptive search intelligence

- `AdaptiveGradientSearch` — Governed `step_fn` combining central-difference gradient estimation with an Adam-style per-dimension adaptive update, Rechenberg 1/5-rule self-adaptive stochastic search, and stagnation-triggered restarts. QES's default intelligent local/global search strategy.
- `AdaptiveSearchConfig` — Configurable hyperparameters for `AdaptiveGradientSearch`: step-size bounds, Adam moment coefficients, gradient probability, and stagnation tuning.
- `OptimizationResult` — Outcome of `optimize()`: best point found, best value, iteration count, evaluation count, and surviving room count.
- `optimize` — End-to-end convenience entry point wiring `RealityGenerator` branching, a permission-gated `QESSpace`, and `AdaptiveGradientSearch` together for any bounded, continuous objective.

### Orchestration and lifecycle

- `Acros` — Self-stabilizing orchestration law applying the correction gradient `xdot = f(x, t) - chi*K*grad[Phi + mu*CCI]` to steer a room toward admissibility.
- `AdaptiveAcros` — `Acros` with an Adam-style per-component adaptive gain replacing the fixed gain `K`, for ill-conditioned objectives.
- `RoomLifecycle` — Validates and applies allowed room lifecycle state transitions across the seven defined lifecycle states.
- `Equation` — Single equation hypothesis carrying parameters, fitness, lineage parent, generation depth, and an optional callable.
- `EquationForge` — SGEE population genetics engine: seeds equations and recursively mutates, crosses over, and selects populations of equation hypotheses.

### Divergence and governance

- `DSA` — Divergence and stability tracker used to evaluate how far a room drifts from its target domain.
- `DivergenceResult` — Structured divergence report containing drift, hazard, and stability data.
- `divergence` — Finite-distance divergence measure used to score admissibility and collapse risk.
- `divergence_gradient` — Gradient of divergence used for local correction and risk-aware exploration.
- `collapse_proximity` — Risk metric combining collapse pressure and proximity to failure states.
- `hsa_state` — Classifies a health score `S_i(t)` into stable/critical/singular based on critical-band tolerance.
- `PermissionClosure` — Filters candidate actions to admissible ones under constraints and closure rules.
- `PermissionClosureResult` — Bundled output of `PermissionClosure.evaluate()`: the admissible action set and a safety flag.
- `action_selection` — Selects the action minimizing `J(x,u) + lambda*S(F(x,u,xi))` from the admissible action set `A(x)`.
- `safe_exploration_closure` — Returns True iff a policy-search operator maps every admissible policy back to an admissible policy.
- `CrossDomainVerifier` — Checks reconstruction, state closure, and constraint closure across domains.
- `DomainVerificationResult` — Verification result: reconstruction error plus state-closure and constraint-closure booleans.

### Advanced architecture

- `Universe` — Top-level recursive digital space containing worlds, agents, and nested universes.
- `World` — Runtime environment hosting rooms, agents, and nested entities.
- `Agent` — Autonomous inhabitant that perceives, decides, and acts within a world or universe.
- `Message` — Single inter-agent message carrying sender id, payload, and timestamp for inbox delivery.
- `PatternMemory` — Stores dominant and non-dominated patterns for reuse across exploration cycles.
- `CosmicVP` — Root hypervisor-style substrate for virtualized resource and control loops.
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
- `necessity_field` — Layer 1: returns True iff `N(x) >= 0`, permitting formation to proceed.
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
- `UniverseTelemetry` — Snapshot of `Q(t)`: time, world count, agent count, equation count, and event count.
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
- `DistributedRuntime` — Deployment-oriented runtime facade extending `ProductionRuntime` with multi-worker orchestration.
- `ApiKeyGuard` — Minimal constant-time HMAC-compared API-key access control for production runtime deployments.
- `RuntimeAccessError` — Permission error raised on API-key validation failure during runtime access.
- `RuntimeTelemetry` — Summary of queue health, completion rates, and runtime performance.

### Validation, safety, and metrics

- `ValidationFramework` — V&V framework: runs a battery of named boolean checks against a state and reports failures.
- `FunctionalSafetySystem` — Scores how far a state sits inside a safety-margin envelope: 1.0 centered, 0.0 at the boundary.
- `IntegrityValidator` — Cross-checks redundant replicas for consistency via componentwise median-consensus voting.
- `FailureRecoveryLoop` — Omega-Loop: error detection -> null baseline -> regeneration -> restabilization within the envelope.
- `convergence_coefficient` — `C_Q = 1 - H_bar_Q`, measuring how far room weights have converged from uniform dispersion.
- `qes_entropy` — Shannon entropy `H_Q = -sum_i p_i * ln(p_i)` of the room-weight distribution.
- `kl_divergence` — Kullback-Leibler divergence `KL(p || q)` measuring drift of current weights from a reference distribution.
- `gini_coefficient` — Gini inequality index of the weight distribution in `[0, 1]`, sensitive to tail concentration.

### Global invariants and event sourcing

- `InvariantEngine` — Recursively checks state validity, dimensional consistency, bounds, lineage integrity, resource consistency, probability normalization, and lifecycle consistency across `Room` -> `QESSpace` -> `World` -> `Universe`; `check(obj)` dispatches on type.
- `InvariantReport` — Aggregate pass/fail result with every `InvariantViolation` found (not just the first); `merge()` combines reports, `summary()` gives a one-line human-readable digest.
- `InvariantViolation` — One failed check: `scope`, `check` name, `message`, and `object_id`.
- `EventLog` — Append-only event-sourcing log; `record(kind, payload, parent_ids, timestamp)` appends an `Event` of one of the canonical `EVENT_KINDS` (`SPAWN`, `EXECUTE`, `DIVERGE`, `PERMISSION`, `MUTATE`, `SELECT`, `MERGE`, `COLLAPSE`).
- `Event` — One recorded state transition; `to_dict()`/`from_dict()` round-trip through JSON-serializable form.
- `LineageGraph` — Parent/child provenance graph over any tracked entity id; `ancestors()`/`descendants()` traverse the graph, `to_dict()`/`from_dict()` serialize it.
- `state_hash` — Deterministic SHA-256 content hash of a state array (shape + dtype + bytes), used to verify bit-identical reproduction.
- `DeterministicReplay` — Records state hashes per step and verifies (`verify`/`verify_all`) that a later replay reproduces them exactly, returning a `ReplayReport` of every `ReplayMismatch`.

### Closed-loop autonomous cycle

- `AutonomousCycle` — Phase 2's outer ring around `QESSpace.step()`: `seed()` registers the initial population, `run_generation()`/`run()` tick the space, extract the dominant room's outcome into `PatternMemory`, check `InvariantEngine.check_space()`, record `SPAWN`/`EXECUTE`/`DIVERGE`/`PERMISSION`/`SELECT`/`MERGE` events, and adaptively branch the next generation.
- `adaptive_branch_count` — `N_children = f(uncertainty, risk, divergence, compute_budget, novelty, historical_success)`; a documented heuristic deciding how many children the next generation gets.
- `GenerationReport` — One generation's outcome: telemetry, invariant report, dominant room id, extracted pattern, next child count, events recorded.
- `CycleHistory` — Accumulated `GenerationReport`s with `convergence_series()`/`entropy_series()` helpers.

### Reality operators and families

- `mutation`, `crossover`, `interpolate`, `extrapolate`, `inversion`, `perturbation` — Deterministic `Room -> Room` operators for structural/state-level branching beyond plain Gaussian noise.
- `dimensional_transformation`, `topology_transformation` — Apply a linear transform or axis permutation to a room's state/bounds/couplings.
- `equation_substitution`, `parameter_transformation` — Replace a room's equation population or transform its parameter dict wholesale.
- `RealityFamily` — A named batch of sibling rooms with `mean_state()`, `std_state()`, `diversity()`, `best(score_fn)`.
- `RealityFamilyBuilder` — Composes several operators (mutation, inversion, and -- given a second parent -- crossover/interpolation/extrapolation) into one `RealityFamily`.

### Equation Forge 2.0 (AST-based equations)

- `EquationAST`, `Constant`, `Variable`, `Operator` — Tree representation of an equation, replacing the flat string form with a structure that can be mutated/crossed-over/simplified node by node.
- `Primitive`, `primitive_library()` — Registered arithmetic/trig/log primitives (`+ - * / ^ log sqrt exp sin cos ...`) with arity and safe (division-by-zero/negative-log guarded) evaluation.
- `simplify_ast`, `subtree_at`, `replace_subtree`, `iter_subtree_paths` — Structural helpers for tree-walking, extraction, and in-place subtree replacement (used by mutation/crossover).
- `DimensionValidation`, `NumericalValidation`, `StabilityReport` — Dimensional-consistency, numerical-domain, and stability checks run against a candidate equation before it's trusted.
- `EquationASTForge`, `EquationASTEvolutionResult`, `EvolutionSnapshot` — Fitness-driven evolution loop that mutates/crosses/simplifies a population of `EquationAST`s toward better fit.

### Reality-to-reality communication

- `CommunicationMessage` (from `qes.communication`, exported under this name to avoid clashing with `qes.agent.Message`), `CommunicationFabric` — In-memory message passing between rooms: `exchange_state`, `exchange_knowledge`, `exchange_pattern`.
- `ResourceBid`, `ResourceNegotiation` — Compute-resource bidding: rooms submit bids with expected utility, `allocate()` distributes a shared budget under a `PermissionClosure`/`GenesisPermission` gate.
- `Coalition`, `form_coalition` — Groups rooms toward a shared goal.
- `compete`, `cooperate` — A pairwise winner-take-more contest and an N-way cooperative aggregate computation.

### Information Gain Engine

- `information_gain(before, after)` — `IG(a) = H(before) - H(after)`, the reduction in `qes.convergence.qes_entropy` a branch is expected to cause.
- `population_entropy(rooms)` — Entropy of a room population's weight distribution.
- `InformationGainEstimator` — Tracks per-candidate entropy-before/after history and reports running information gain.
- `rank_by_information_gain`, `allocate_by_information_gain` — Rank candidates by expected information gain and allocate a shared compute budget proportional to it.

### Persistent Knowledge Graph

- `KnowledgeNode`, `NODE_KINDS` — One typed node (`reality`/`observation`/`equation`/`pattern`/`result`/`causal_relation`) with a JSON-serializable payload and optional success/failure label.
- `KnowledgeGraph` — Typed provenance graph: `add_node`/`get`/`nodes`/`ancestors`/`descendants` for structure, `record_experiment`/`successful_configurations`/`failed_configurations` for outcome tracking, `record_causal_hypothesis`/`strengthen_hypothesis`/`causal_hypotheses_for` for cause-effect confidence, `add_counterexample`/`counterexamples` and `add_domain_relationship`/`domain_relationships` for cross-cutting metadata, `to_dict`/`from_dict`/`save`/`load` for JSON persistence.
- `CausalHypothesis` — A `cause -> effect` hypothesis with a confidence in `[0, 1]` that strengthens/weakens with evidence.

### Causal Reality Engine

- `CausalGraph`, `CausalEdge` — Named-variable DAG plus weighted `cause -> effect` links used to define the causal structure for a domain.
- `intervene`, `InterventionResult` — Apply a `do(X=x)` change to a baseline room/state and report the propagated state and outcome deltas.
- `counterfactual`, `CounterfactualResult` — Compare the actual observed state against a hypothetical alternate value for one variable.
- `perturbation_sensitivity`, `SensitivityResult` — Scan each variable with repeated small perturbations and rank the largest outcome sensitivities.
- `hidden_variable_hypotheses`, `HiddenVariableHypothesis` — Flag strong unexplained correlations as candidate latent-variable explanations.
- `analyze_causality`, `CausalDiscoveryResult` — Run the full intervention/counterfactual/sensitivity/hidden-variable pass and return one bundled result.

### Reality Marketplace

- `RealityAccount` — Per-reality market state: hard compute/memory/energy budgets plus priority, risk, expected utility, novelty, and historical performance.
- `MarketBid` — One multi-resource request with uncertainty and optional utility/risk/novelty overrides; `from_entropy_change()` derives its information-gain term from Phase 8.
- `compute_allocation_score` — Transparent weighted score `f(utility, risk, novelty, uncertainty, information_gain)` used to rank bids before budget splitting.
- `RealityMarketplace` — Registers accounts, queues one bid per account per round, clears resource pools, tracks standings, and updates historical performance.

### Digital Twin 2.0

- `Observation` — One timestamped sensor observation with optional uncertainty and sensor id.
- `sensor_fusion` — Inverse-variance fusion of several noisy observations into one fused reading.
- `OnlineParameterEstimator` — Recursive least-squares estimator for online linear parameter updates.
- `StateEstimator` — Classical linear Kalman filter with predict/update steps.
- `uncertainty_propagation`, `PropagationResult` — Finite-difference covariance propagation through a transition function.
- `ModelSelector`, `ModelSelectionResult` — Score and rank candidate predictive models by residual fit over recent history.
- `anomaly_detection`, `AnomalyReport` — Residual-based anomaly screen using per-channel z-scores.
- `prediction_interval`, `calibration_check` — Gaussian interval construction plus empirical coverage measurement over realized outcomes.
- `drift_detection`, `DriftReport` — Rolling-window residual-drift diagnostic comparing recent behavior against an older baseline.
- `DigitalTwinLoop`, `LoopStepResult` — End-to-end observe/update/predict/action orchestrator returning one complete loop-step report.

### Accelerator backends and distributed coordination

- `ComputeBackend` — Abstract backend interface for array creation, matmul, elementwise operations, NumPy conversion, and optional synchronization.
- `NumpyBackend`, `GPUBackend` — Guaranteed CPU backend and honest optional GPU backend that activates only when a real accelerator runtime exists.
- `BackendRegistry`, `get_backend`, `benchmark_backend`, `BenchmarkResult` — Backend factory/lookup helpers plus repeatable wall-clock operation benchmarking.
- `WorkItem` — One checkpoint-serializable distributed task payload, typically a `Room` or room-like object.
- `Worker` — Thread-backed simulated worker that heartbeats, processes leased work, and can be stopped or crashed for testing.
- `Controller` — Central scheduler handling worker registration, leases, results, failures, heartbeats, recovery, checkpointing, and GPU-aware placement.
- `distributed_checkpoint`, `restore_from_checkpoint` — Serialize/restore the controller state as a plain dict checkpoint.
- `autoscaling_policy` — Heuristic worker-count recommendation from outstanding work and current utilization.

### Meta-learning

- `ProblemStructure` — Cheap feature vector describing a search problem before a run starts.
- `SearchStrategy` — Complete optimizer/hyperparameter recommendation produced by the meta-controller.
- `SearchOutcome` — One completed run stored as evidence for later strategy recommendations.
- `SearchHistory` — Local and optional `PatternMemory`-backed collection of prior outcomes.
- `MetaController` — Transparent heuristic recommender that blends problem features with similar successful history.
- `record_outcome` — Append one run result to `SearchHistory` and optionally mirror it into `PatternMemory`.

### Self-improvement

- `PerformanceEvaluation` — Measured score / convergence-speed / resource-cost summary for one current strategy.
- `ArchitectureProposal` — Data-only alternative strategy/configuration proposed for evaluation; never a source-code patch.
- `BenchmarkResult`, `SafetyTestResult`, `RegressionTestResult`, `ProposalEvaluation` — Structured outputs of the sandbox, safety, regression, and final decision stages.
- `Sandbox`, `run_in_sandbox` — Benchmark one proposal on a cloned copy of its strategy configuration.
- `run_safety_tests`, `run_regression_tests` — Apply caller-supplied guardrails and baseline comparisons to one proposal.
- `ApprovalPolicy`, `default_approval_policy` — Deterministic final approval gate that stands in for a human/policy decision in tests and demos.
- `SelfImprovementPipeline`, `SelfImprovementReport`, `propose_alternatives` — Full proposal -> benchmark -> safety -> regression -> approval -> in-memory deployment loop.

### Formal verification

- `VerificationStageResult` — One stage outcome with pass/fail/skipped status, message, and structured details.
- `numerical_verification` — Check finite values, magnitudes, and optional equation numerical validity over sample batches.
- `constraint_verification` — Verify that a candidate state respects supplied lower/upper bounds.
- `safety_verification` — Push a candidate through a `GenesisPermission`/`AdaptivePermission`-style gate and normalize the result.
- `stability_verification` — Run a perturbation-based stability screen or equation stability analysis.
- `invariant_verification` — Wrap `InvariantEngine.check(candidate)` as one verification stage.
- `cross_domain_verification` — Apply `CrossDomainVerifier` across one or more domain payloads and aggregate the outcome.
- `Evidence` — Ordered evidence package containing every verification stage and the overall pass/fail decision.
- `VerificationEngine` — Run the full staged verification chain and return `Evidence(candidate)`.

### Adversarial testing

- `FailureCategory` — Enumerated failure families such as boundary failure, unstable state, singularity, resource exhaustion, and cascading failure.
- `AdversarialCandidate` — One deliberately stressful room plus rationale, category, and stress score.
- `AdversarialGenerator` — Deterministic generator for boundary, unstable-state, singularity, adversarial-input, resource, equation, coupling, and cascading probes.
- `FailureAnalysis` — Outcome of evaluating one adversarial candidate against permission and optional verification logic.
- `RedTeamReport`, `run_red_team_campaign` — Aggregate a full red-team campaign with category counts, failure rates, and suggested improvements.

### Multi-agent evolution

- `SpecializedAgent` — One lightweight heuristic scorer for a specific specialty such as optimization, safety, prediction, or causal analysis.
- `AgentPopulation` — Run competition, cooperation, knowledge exchange, coalition formation, and final collective choice across specialists.
- `ScoredCandidate`, `AgentPerformanceRecord`, `SharedInsight`, `Coalition` — Structured records for rankings, per-round agent performance, exchanged insights, and small alliances.
- `CollectiveSolutionResult`, `collective_solution` — End-to-end population choice: rankings, shared insights, coalitions, aggregate scores, and the selected winner.
- `ComparisonResult`, `compare_individual_vs_collective` — Compare the collective winner against each specialist's individually preferred candidate.

### Benchmarking

- `BenchmarkProblem` — Bounded objective definition with optional known optimum and constraint violation scoring.
- `make_sphere_problem`, `make_rastrigin_problem`, `make_rosenbrock_problem` — Constructors for the built-in benchmark problems.
- `BaselineResult`, `BenchmarkMetrics`, `BenchReport` — Per-run result, aggregated metric bundle, and full report/table wrapper for a benchmark campaign.
- `run_qes_baseline` — Run the real `qes.intelligence.optimize` entry point under an exact evaluation budget.
- `run_bayesian_optimization_baseline`, `run_evolutionary_algorithm_baseline`, `run_genetic_algorithm_baseline`, `run_simulated_annealing_baseline`, `run_cma_es_baseline`, `run_differential_evolution_baseline`, `run_gradient_method_baseline`, `run_mpc_baseline`, `run_reinforcement_learning_baseline`, `run_symbolic_regression_baseline`, `run_multi_objective_baseline` — Honestly labeled minimal reference baselines used for fair comparison.
- `BASELINE_REGISTRY`, `DEFAULT_BENCHMARK_PROBLEMS`, `SPHERE_PROBLEM`, `RASTRIGIN_PROBLEM`, `ROSENBROCK_PROBLEM`, `SCIPY_AVAILABLE`, `run_qes_bench` — Benchmark campaign registry/constants and the top-level runner.

### Reproducibility

- `ExperimentArtifact` — Complete persisted artifact: code version, configuration, seeds, hardware, population, lineage, equations, decisions, allocations, results, failures, and final evidence.
- `ExperimentRecorder` — Incremental recorder that assembles an `ExperimentArtifact` across a run.
- `save_artifact`, `load_artifact` — JSON persistence helpers for experiment artifacts.
- `reproduce_experiment` — Re-run a saved experiment with the same recorded configuration and seeds.
- `ReproductionCheckResult`, `verify_reproduction` — Compare original and reproduced artifacts within a numeric tolerance.

### Observatory

- `ObservatorySnapshot` — Renderable observatory state containing headline metrics, the reality tree, and optional module summaries.
- `render_reality_tree` — Convert a parent/child adjacency map into a box-drawing ASCII/Unicode tree.
- `ObservatoryDashboard` — Stateful dependency-free renderer for observatory snapshots, including `render_live()` for successive updates.
- `build_snapshot_from_modules` — Assemble a snapshot from whichever divergence, permission, marketplace, digital-twin, failure, pattern, or knowledge-graph outputs are available.

### Domain packs

- `ControlPolicyPack`, `ControlPolicyPackConfig`, `ControlPolicyPackResult`, `make_control_policy_space` — Validated adapter that configures a `QESSpace` to search controller-gain space for a policy driving a bounded plant toward a target state.
- `DesignSpacePack`, `DesignSpacePackConfig`, `DesignSpacePackResult`, `make_design_space` — Validated adapter that configures a `QESSpace` to search a constrained structural-design parameter space for a low-cost, admissible design.
- `ResourceAllocationPack`, `ResourceAllocationPackConfig`, `ResourceAllocationPackResult`, `ResourceJob`, `make_resource_allocation_pack` — Validated adapter over `qes.compute_allocator` that computes a priority-weighted allocation of a shared budget across a set of jobs.
- `DomainPackDescriptor`, `DomainPackRegistry`, `list_domain_packs` — Discoverability helpers enumerating the available domain packs by name.

### QES SDK

- `QESClient` — High-level orchestration client: given state bounds, a step function, and optional permission configuration, wires up `RealityGenerator` + `QESSpace` + `GenesisPermission`/`AdaptivePermission` + `PatternMemory` and exposes `.run(steps=N)`.
- `SDKPermissionConfig` — Configuration dataclass for the permission kernel a `QESClient` should use.
- `SDKRunResult` — Result of a `QESClient.run()` call: best room/state, final entropy/convergence, wall time, and step count.
- `quick_search` — One-shot convenience function wrapping `QESClient` for the simplest bounded-search use case.

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

`optimize()` runs QES's adaptive gradient search (Adam-style per-dimension
steps + self-adaptive stochastic search + stagnation restarts) through the
full governed room/space loop, and is competitive with derivative-free
external baselines like SciPy's Nelder-Mead on standard benchmarks such as
the Rosenbrock function (see `examples/scipy_baseline_benchmark.py`).

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
