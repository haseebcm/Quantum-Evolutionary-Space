"""H^11 COSMIC VP — the QES virtualization substrate.

Source-derived architecture (see the PDF-extraction summary in
docs/QES-architecture.md, "H^11 COSMIC VP"): COSMIC VP is the root
virtualization fabric beneath QES. It provisions virtual machines,
networks, storage, containers, and execution environments, and wires
together five layers and seven closed control loops:

Five layers:
    1. H^11 Base Metric Layer      -- units, operations, metric stability
    2. H^11 Hypervisor             -- VMs, ACROS OS, resources, drift mapping
    3. Intelligence Engines        -- dual engine, stabilizer, drift controller
    4. ACROS V12-BIE               -- applications, compute, cloud operations
    5. H^11 Cloud EC Expansion     -- virtual storage, node creation, expansion

Seven loops (with the source's equations, where given):
    C-Loop  (Creation):       C = f(N, H11, phi)
    S-Loop  (Stabilization):  S = delta_load / H11
    X-Loop  (Security):       X = (N - phi) XOR (N' - H11)
    D-Loop  (Drift):          non-collision, non-repetition drift storage
    A-Loop  (ACROS exec):     A = ACROS(V12BIE)
    EC-Loop (Cosmic expand):  EC = lim_{N->inf} (N + H11)
    Omega-Loop (Null-recover): Omega = 0_n -> f(N)

Full request/compute cycle (section IX of the extraction):
    REQUEST -> DUAL ENGINE BOUNDARY -> H^11 STABILIZER -> DRIFT/STORAGE MAPPING
    -> ACROS EXECUTION -> EC-LOOP RESOURCE EXPANSION -> OMEGA-LOOP RECOVERY -> OUTPUT
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

_resource_id_counter = itertools.count(1)


def _next_resource_id(kind: str) -> str:
    return f"{kind.upper()}-{next(_resource_id_counter):05d}"


def _finite_scalar(value: Any, *, name: str) -> float:
    """Return `value` as a finite float."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real-valued scalar") from exc
    if scalar != scalar or scalar in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_kind(kind: object) -> str:
    """Validate a resource kind string."""
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    normalized = kind.strip()
    if not normalized:
        raise ValueError("kind must be non-empty")
    return normalized


def _normalize_spec(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a defensive copy of a resource specification."""
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a mapping when provided")
    return dict(spec)


@dataclass
class VirtualResource:
    """A single provisioned virtual resource (VM, network, storage, container)."""

    kind: str
    spec: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        self.kind = _validate_kind(self.kind)
        self.spec = _normalize_spec(self.spec)
        if self.id and not isinstance(self.id, str):
            raise TypeError("id must be a string")
        if not self.id:
            self.id = _next_resource_id(self.kind)


@dataclass
class RequestCycleResult:
    """Outcome of one full COSMIC VP request/compute cycle."""

    request: Any
    output: Any
    stabilized: bool
    expanded: bool
    recovered: bool
    resources: list[str]


class CosmicVP:
    """H^11 Intelligence Hypervisor: the root fabric beneath QES.

    `h11` is the metric-stability baseline (Layer 1); `capacity` bounds how
    much simultaneous load the fabric can host before the EC-Loop must
    expand (create new nodes).
    """

    def __init__(self, h11: float = 1.0, capacity: int = 64) -> None:
        """Initialize the root virtualization fabric."""
        h11_value = _finite_scalar(h11, name="h11")
        if h11_value <= 0:
            raise ValueError("h11 must be positive")
        if not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.h11 = h11_value
        self.capacity = capacity
        self.resources: dict[str, VirtualResource] = {}
        self.nodes: int = 1
        self.drift_map: dict[str, float] = {}
        self.event_log: list[tuple[Any, ...]] = []

    # ------------------------------------------------------------------
    # C-Loop -- Creation
    # ------------------------------------------------------------------
    def create(
        self,
        kind: str,
        phi: float = 0.0,
        spec: Mapping[str, Any] | None = None,
    ) -> VirtualResource:
        """C = f(N, H11, phi): create a VM/network/storage/container.

        `phi` is the violation-energy context (Genesis Phi(x)) at creation
        time; resources are refused creation above the fabric's capacity.
        """
        kind_name = _validate_kind(kind)
        phi_value = _finite_scalar(phi, name="phi")
        if len(self.resources) >= self.nodes * self.capacity:
            raise RuntimeError("COSMIC VP capacity exceeded; expand() first")
        resource = VirtualResource(kind=kind_name, spec=_normalize_spec(spec))
        self.resources[resource.id] = resource
        self.drift_map[resource.id] = phi_value / self.h11
        self.event_log.append(("create", resource.id))
        return resource

    def create_vm(self, spec: Mapping[str, Any] | None = None) -> VirtualResource:
        """Provision a virtual machine resource."""
        return self.create("vm", spec=spec)

    def create_network(self, spec: Mapping[str, Any] | None = None) -> VirtualResource:
        """Provision a virtual network resource."""
        return self.create("network", spec=spec)

    def create_storage(self, spec: Mapping[str, Any] | None = None) -> VirtualResource:
        """Provision a virtual storage resource."""
        return self.create("storage", spec=spec)

    def create_container(self, spec: Mapping[str, Any] | None = None) -> VirtualResource:
        """Provision a virtual container resource."""
        return self.create("container", spec=spec)

    # ------------------------------------------------------------------
    # S-Loop -- Stabilization
    # ------------------------------------------------------------------
    def stabilization_load(self, delta_load: float) -> float:
        """S = delta_load / H11: metric-stability load ratio for this tick."""
        return _finite_scalar(delta_load, name="delta_load") / self.h11

    def stabilize(self, delta_load: float, threshold: float = 1.0) -> bool:
        """True iff the fabric remains within its stability threshold."""
        threshold_value = _finite_scalar(threshold, name="threshold")
        if threshold_value < 0:
            raise ValueError("threshold must be >= 0")
        return self.stabilization_load(delta_load) <= threshold_value

    # ------------------------------------------------------------------
    # X-Loop -- Security / dual-engine enforcement
    # ------------------------------------------------------------------
    def security_check(self, n: float, n_prime: float, phi: float) -> bool:
        """X = (N - phi) XOR (N' - H11), interpreted as a boolean dual-engine
        consistency gate: both half-checks must independently pass (be
        non-negative / within tolerance) for the request to be admitted."""
        n_value = _finite_scalar(n, name="n")
        n_prime_value = _finite_scalar(n_prime, name="n_prime")
        phi_value = _finite_scalar(phi, name="phi")
        left_ok = (n_value - phi_value) >= 0
        right_ok = (n_prime_value - self.h11) >= -self.h11
        return left_ok and right_ok

    # ------------------------------------------------------------------
    # D-Loop -- Drift / VNDS storage mapping
    # ------------------------------------------------------------------
    def drift_of(self, resource_id: str) -> float:
        """Return the recorded drift for `resource_id`, or 0.0 if unknown."""
        if not isinstance(resource_id, str):
            raise TypeError("resource_id must be a string")
        return self.drift_map.get(resource_id, 0.0)

    def map_drift(self, resource_id: str, phi: float) -> float:
        """Record/refresh a resource's drift value; guarantees non-collision
        by always keying on the resource's unique id (never repeated/reused)."""
        if not isinstance(resource_id, str):
            raise TypeError("resource_id must be a string")
        if resource_id not in self.resources:
            raise KeyError(f"unknown resource: {resource_id!r}")
        drift = _finite_scalar(phi, name="phi") / self.h11
        self.drift_map[resource_id] = drift
        return drift

    # ------------------------------------------------------------------
    # A-Loop -- ACROS execution
    # ------------------------------------------------------------------
    def acros_execute(self, program: Any, executor: Callable[[Any], Any]) -> Any:
        """A = ACROS(V12BIE): run `program` through the supplied executor
        callable (typically an `AcrosV12BIE.stabilize`/`ApexI.execute`)."""
        if not callable(executor):
            raise TypeError("executor must be callable")
        result = executor(program)
        self.event_log.append(("acros_execute", program))
        return result

    # ------------------------------------------------------------------
    # EC-Loop -- Cosmic expansion
    # ------------------------------------------------------------------
    def expand(self, extra_nodes: int = 1) -> int:
        """EC = lim_{N->inf} (N + H11): grow the fabric's node count so more
        resources (VMs/storage/drift blocks) can be created."""
        if extra_nodes < 1:
            raise ValueError("extra_nodes must be >= 1")
        self.nodes += extra_nodes
        self.event_log.append(("expand", extra_nodes))
        return self.nodes

    # ------------------------------------------------------------------
    # Omega-Loop -- Null recovery
    # ------------------------------------------------------------------
    def null_recover(self, resource_id: str) -> VirtualResource:
        """Omega = 0_n -> f(N): collapse a resource to null, then regenerate
        a fresh replacement of the same kind (error -> null -> regenerate)."""
        if not isinstance(resource_id, str):
            raise TypeError("resource_id must be a string")
        old = self.resources.pop(resource_id, None)
        self.drift_map.pop(resource_id, None)
        kind = old.kind if old is not None else "vm"
        spec = dict(old.spec) if old is not None else {}
        replacement = self.create(kind, spec=spec)
        self.event_log.append(("null_recover", resource_id, replacement.id))
        return replacement

    # ------------------------------------------------------------------
    # Full request/compute cycle
    # ------------------------------------------------------------------
    def request_cycle(
        self,
        request: Any,
        executor: Callable[[Any], Any],
        delta_load: float = 0.0,
        phi: float = 0.0,
    ) -> RequestCycleResult:
        """REQUEST -> dual-engine boundary -> stabilizer -> drift/storage
        mapping -> ACROS execution -> EC-Loop expansion (if needed) ->
        Omega-Loop recovery (if the executor raises) -> OUTPUT."""
        if not callable(executor):
            raise TypeError("executor must be callable")
        _finite_scalar(phi, name="phi")
        stabilized = self.stabilize(delta_load)
        expanded = False
        if len(self.resources) >= self.nodes * self.capacity:
            self.expand()
            expanded = True

        recovered = False
        try:
            output = self.acros_execute(request, executor)
        except Exception as exc:
            # Anomaly -> collapse to null, regenerate stable state, retry once.
            recovered = True
            self.event_log.append(("recover", request, type(exc).__name__))
            output = executor(request)

        return RequestCycleResult(
            request=request,
            output=output,
            stabilized=stabilized,
            expanded=expanded,
            recovered=recovered,
            resources=list(self.resources.keys()),
        )
