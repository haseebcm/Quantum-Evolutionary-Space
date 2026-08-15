"""Phase 19 -- Reproducibility infrastructure.

Every experiment should produce a complete artifact:

    Experiment ID
    ├── code version
    ├── configuration
    ├── random seeds
    ├── hardware
    ├── population
    ├── lineage
    ├── equations
    ├── decisions
    ├── resource allocation
    ├── results
    ├── failures
    └── final evidence

This module implements that artifact as plain Python dataclasses plus JSON
save/load helpers. The "hardware" capture is honest local-machine metadata
from Python's `platform` module, and reproduction remains ordinary
deterministic Python/NumPy execution, not literal quantum hardware.
"""
from __future__ import annotations

import json
import platform
import random
import subprocess
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qes.knowledge_graph import KnowledgeGraph
    from qes.verification import Evidence

try:
    from qes import __version__ as _QES_VERSION
except ImportError:  # pragma: no cover - exercised only when package import fails.
    _QES_VERSION = "unknown"

try:
    from qes.knowledge_graph import KnowledgeGraph

    _KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when module is absent.
    KnowledgeGraph = Any  # type: ignore[misc,assignment]
    _KNOWLEDGE_GRAPH_AVAILABLE = False

try:
    from qes.verification import Evidence

    _EVIDENCE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when module is absent.
    Evidence = Any  # type: ignore[misc,assignment]
    _EVIDENCE_AVAILABLE = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_ready(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable plain Python objects."""
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if _EVIDENCE_AVAILABLE and isinstance(value, Evidence):
        stages = [
            {
                "stage": stage.stage,
                "passed": stage.passed,
                "message": stage.message,
                "details": _json_ready(stage.details),
                "skipped": stage.skipped,
                "status": stage.status,
            }
            for stage in value.stages
        ]
        return {
            "evidence_type": "qes.verification.Evidence",
            "candidate_repr": repr(value.candidate),
            "passed": value.passed,
            "summary": value.summary,
            "stages": stages,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_ready(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalized_mapping(
    value: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if value is not None:
        if not isinstance(value, Mapping):
            raise TypeError("expected a mapping")
        data.update({str(key): item for key, item in value.items()})
    data.update({str(key): item for key, item in kwargs.items()})
    return {key: _json_ready(item) for key, item in data.items()}


def _capture_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def _capture_code_version() -> dict[str, str]:
    return {"qes_version": str(_QES_VERSION), "git_commit": _capture_git_commit()}


def _capture_hardware() -> dict[str, str]:
    return {
        "label": "local machine info",
        "system": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
    }


def _seed_value(random_seeds: dict[str, Any] | list[Any]) -> int | None:
    if isinstance(random_seeds, Mapping):
        for key in ("numpy", "np", "seed", "rng_seed", "random_seed"):
            value = random_seeds.get(key)
            if isinstance(value, Real) and not isinstance(value, bool):
                return int(float(value))
        for value in random_seeds.values():
            if isinstance(value, Real) and not isinstance(value, bool):
                return int(float(value))
        return None
    for value in random_seeds:
        if isinstance(value, Real) and not isinstance(value, bool):
            return int(float(value))
    return None


def _python_seed_value(random_seeds: dict[str, Any] | list[Any]) -> int | None:
    if isinstance(random_seeds, Mapping):
        for key in ("python", "py", "random"):
            value = random_seeds.get(key)
            if isinstance(value, Real) and not isinstance(value, bool):
                return int(float(value))
    return _seed_value(random_seeds)


def _load_artifact_input(artifact_or_path: ExperimentArtifact | str | Path) -> ExperimentArtifact:
    if isinstance(artifact_or_path, ExperimentArtifact):
        return artifact_or_path
    return load_artifact(artifact_or_path)


def _compare_values(
    left: Any,
    right: Any,
    *,
    tolerance: float,
    prefix: str,
    differing_keys: list[str],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        all_keys = sorted({str(key) for key in left} | {str(key) for key in right})
        for key in all_keys:
            next_prefix = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differing_keys.append(next_prefix)
                continue
            _compare_values(
                left[key],
                right[key],
                tolerance=tolerance,
                prefix=next_prefix,
                differing_keys=differing_keys,
            )
        return

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differing_keys.append(prefix)
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            next_prefix = f"{prefix}[{index}]"
            _compare_values(
                left_item,
                right_item,
                tolerance=tolerance,
                prefix=next_prefix,
                differing_keys=differing_keys,
            )
        return

    if (
        isinstance(left, Real)
        and isinstance(right, Real)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        if not bool(np.isclose(float(left), float(right), atol=tolerance, rtol=0.0)):
            differing_keys.append(prefix)
        return

    if left != right:
        differing_keys.append(prefix)


@dataclass
class ExperimentArtifact:
    """Complete persisted artifact for one experiment run."""

    experiment_id: str
    code_version: dict[str, Any]
    configuration: dict[str, Any]
    random_seeds: dict[str, Any] | list[Any]
    hardware: dict[str, Any]
    population: dict[str, Any] | list[Any] = field(default_factory=dict)
    lineage: dict[str, Any] | list[Any] = field(default_factory=list)
    equations: list[Any] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    resource_allocation: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    final_evidence: dict[str, Any] | list[Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("experiment_id must be a non-empty string")
        if not isinstance(self.code_version, dict):
            raise TypeError("code_version must be a dict")
        if not isinstance(self.configuration, dict):
            raise TypeError("configuration must be a dict")
        if not isinstance(self.random_seeds, (dict, list)):
            raise TypeError("random_seeds must be a dict or list")
        if not isinstance(self.hardware, dict):
            raise TypeError("hardware must be a dict")
        if not isinstance(self.population, (dict, list)):
            raise TypeError("population must be a dict or list")
        if not isinstance(self.lineage, (dict, list)):
            raise TypeError("lineage must be a dict or list")
        if not isinstance(self.equations, list):
            raise TypeError("equations must be a list")
        if not isinstance(self.decisions, list):
            raise TypeError("decisions must be a list")
        if not isinstance(self.resource_allocation, dict):
            raise TypeError("resource_allocation must be a dict")
        if not isinstance(self.results, dict):
            raise TypeError("results must be a dict")
        if not isinstance(self.failures, list):
            raise TypeError("failures must be a list")
        if self.final_evidence is not None and not isinstance(self.final_evidence, (dict, list)):
            raise TypeError("final_evidence must be a dict, list, or None")

        self.code_version = _json_ready(self.code_version)
        self.configuration = _json_ready(self.configuration)
        self.random_seeds = _json_ready(self.random_seeds)
        self.hardware = _json_ready(self.hardware)
        self.population = _json_ready(self.population)
        self.lineage = _json_ready(self.lineage)
        self.equations = _json_ready(self.equations)
        self.decisions = _json_ready(self.decisions)
        self.resource_allocation = _json_ready(self.resource_allocation)
        self.results = _json_ready(self.results)
        self.failures = _json_ready(self.failures)
        self.final_evidence = _json_ready(self.final_evidence)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the artifact."""
        return {
            "experiment_id": self.experiment_id,
            "code_version": self.code_version,
            "configuration": self.configuration,
            "random_seeds": self.random_seeds,
            "hardware": self.hardware,
            "population": self.population,
            "lineage": self.lineage,
            "equations": self.equations,
            "decisions": self.decisions,
            "resource_allocation": self.resource_allocation,
            "results": self.results,
            "failures": self.failures,
            "final_evidence": self.final_evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentArtifact:
        """Reconstruct an artifact previously serialized with `to_dict()`."""
        required = {
            "experiment_id",
            "code_version",
            "configuration",
            "random_seeds",
            "hardware",
            "population",
            "lineage",
            "equations",
            "decisions",
            "resource_allocation",
            "results",
            "failures",
            "final_evidence",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"artifact data is missing required fields: {missing}")
        return cls(
            experiment_id=str(data["experiment_id"]),
            code_version=dict(data["code_version"]),
            configuration=dict(data["configuration"]),
            random_seeds=data["random_seeds"],
            hardware=dict(data["hardware"]),
            population=data["population"],
            lineage=data["lineage"],
            equations=list(data["equations"]),
            decisions=[dict(item) for item in data["decisions"]],
            resource_allocation=dict(data["resource_allocation"]),
            results=dict(data["results"]),
            failures=[dict(item) for item in data["failures"]],
            final_evidence=data["final_evidence"],
        )


@dataclass(frozen=True)
class ReproductionCheckResult:
    """Outcome of comparing original and reproduced experiment results."""

    matches: bool
    tolerance: float
    differing_keys: list[str]
    compared_keys: list[str]

    def __str__(self) -> str:
        if self.matches:
            return f"match within tolerance={self.tolerance:g} for keys={self.compared_keys}"
        return (
            f"mismatch within tolerance={self.tolerance:g}; differing_keys={self.differing_keys}"
        )


class ExperimentRecorder:
    """Incremental recorder that assembles a complete `ExperimentArtifact`."""

    def __init__(self, experiment_id: str | None = None) -> None:
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self._configuration: dict[str, Any] = {}
        self._random_seeds: dict[str, Any] = {}
        self._population: dict[str, Any] | list[Any] = {}
        self._lineage: list[dict[str, Any]] = []
        self._equations: list[Any] = []
        self._decisions: list[dict[str, Any]] = []
        self._resource_allocation: dict[str, Any] = {}
        self._results: dict[str, Any] = {}
        self._failures: list[dict[str, Any]] = []
        self._final_evidence: dict[str, Any] | list[Any] | None = None
        self._knowledge_graph: KnowledgeGraph | None = (
            KnowledgeGraph() if _KNOWLEDGE_GRAPH_AVAILABLE else None
        )

    def __enter__(self) -> ExperimentRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc_type is not None and exc is not None:
            self.record_failure(
                exc,
                traceback="".join(traceback.format_exception(exc_type, exc, tb)),
                context="context_manager_exit",
            )

    def record_configuration(self, configuration: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Merge configuration values into the recorder."""
        self._configuration.update(_normalized_mapping(configuration, **kwargs))

    def record_seed(
        self,
        name: str | Mapping[str, Any] | None = None,
        value: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Record one or more random seeds used by the experiment."""
        if isinstance(name, Mapping):
            self._random_seeds.update(_normalized_mapping(name, **kwargs))
            return
        if name is not None:
            if not isinstance(name, str) or not name:
                raise ValueError("seed name must be a non-empty string")
            self._random_seeds[str(name)] = _json_ready(value)
        if kwargs:
            self._random_seeds.update(_normalized_mapping(**kwargs))
        if name is None and value is not None and not kwargs:
            raise ValueError("record_seed requires a seed name when value is provided")

    def record_population(self, population: Mapping[str, Any] | list[Any] | np.ndarray) -> None:
        """Store a snapshot of the current population/state."""
        if not isinstance(population, (Mapping, list, np.ndarray)):
            raise TypeError("population must be a mapping, list, or numpy array")
        self._population = _json_ready(population)

    def record_lineage_step(self, step: str | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Append one lineage/provenance entry."""
        if isinstance(step, str):
            entry = _normalized_mapping({"step": step}, **kwargs)
        else:
            entry = _normalized_mapping(step, **kwargs)
        if not entry:
            raise ValueError("lineage step must not be empty")
        self._lineage.append(entry)

    def record_equation(self, equation: Any) -> None:
        """Append an equation/expression record."""
        self._equations.append(_json_ready(equation))

    def record_decision(self, decision: str | Mapping[str, Any], **kwargs: Any) -> None:
        """Append one decision-log entry."""
        if isinstance(decision, str):
            entry = {"decision": decision}
        elif isinstance(decision, Mapping):
            entry = {str(key): value for key, value in decision.items()}
        else:
            raise TypeError("decision must be a string or mapping")
        entry.update(kwargs)
        self._decisions.append(_json_ready(entry))

    def record_resource_allocation(
        self,
        allocation: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Merge resource-budget or counter information into the recorder."""
        self._resource_allocation.update(_normalized_mapping(allocation, **kwargs))

    def record_result(
        self,
        result: str | Mapping[str, Any],
        value: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Merge final result values into the recorder."""
        if isinstance(result, str):
            if not result:
                raise ValueError("result key must be a non-empty string")
            payload: dict[str, Any] = {result: value}
        elif isinstance(result, Mapping):
            payload = {str(key): item for key, item in result.items()}
        else:
            raise TypeError("result must be a string key or mapping")
        payload.update(kwargs)
        self._results.update(_json_ready(payload))

    def record_failure(
        self,
        failure: BaseException | str | Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        """Append one failure or error record."""
        if isinstance(failure, BaseException):
            entry: dict[str, Any] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        elif isinstance(failure, str):
            entry = {"message": failure}
        elif isinstance(failure, Mapping):
            entry = {str(key): value for key, value in failure.items()}
        else:
            raise TypeError("failure must be an exception, string, or mapping")
        entry.update(kwargs)
        self._failures.append(_json_ready(entry))

    def record_evidence(self, evidence: Any) -> None:
        """Store the final evidence package or summary."""
        self._final_evidence = _json_ready(evidence)

    def finalize(self) -> ExperimentArtifact:
        """Assemble and return the complete experiment artifact."""
        lineage: dict[str, Any] | list[Any]
        if self._knowledge_graph is not None and len(self._knowledge_graph) > 0:
            lineage = {
                "entries": list(self._lineage),
                "knowledge_graph": self._knowledge_graph.to_dict(),
            }
        else:
            lineage = list(self._lineage)
        return ExperimentArtifact(
            experiment_id=self.experiment_id,
            code_version=_capture_code_version(),
            configuration=dict(self._configuration),
            random_seeds=dict(self._random_seeds),
            hardware=_capture_hardware(),
            population=self._population,
            lineage=lineage,
            equations=list(self._equations),
            decisions=list(self._decisions),
            resource_allocation=dict(self._resource_allocation),
            results=dict(self._results),
            failures=list(self._failures),
            final_evidence=self._final_evidence,
        )


def save_artifact(artifact: ExperimentArtifact, path: str | Path) -> None:
    """Persist an artifact to a JSON file."""
    if not isinstance(artifact, ExperimentArtifact):
        raise TypeError("artifact must be an ExperimentArtifact")
    Path(path).write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")


def load_artifact(path: str | Path) -> ExperimentArtifact:
    """Load an artifact previously saved via `save_artifact()`."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"artifact file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("artifact JSON root must be an object")
    return ExperimentArtifact.from_dict(data)


def reproduce_experiment(
    artifact_or_path: ExperimentArtifact | str | Path,
    run_fn: Callable[[dict[str, Any], dict[str, Any] | list[Any]], Mapping[str, Any]],
) -> ExperimentArtifact:
    """Re-run an experiment from a saved artifact using the same config and seeds."""
    if not callable(run_fn):
        raise TypeError("run_fn must be callable")

    original = _load_artifact_input(artifact_or_path)
    recorder = ExperimentRecorder()
    recorder.record_configuration(original.configuration)
    if isinstance(original.random_seeds, dict):
        recorder.record_seed(original.random_seeds)
        invocation_seeds: dict[str, Any] | list[Any] = dict(original.random_seeds)
    else:
        invocation_seeds = list(original.random_seeds)

    recorder.record_lineage_step(
        action="reproduction",
        source_experiment_id=original.experiment_id,
        command="reproduce_experiment",
        knowledge_graph_available=_KNOWLEDGE_GRAPH_AVAILABLE,
    )
    recorder.record_decision(
        "Reproduced experiment from saved artifact",
        source_experiment_id=original.experiment_id,
    )

    numpy_seed = _seed_value(original.random_seeds)
    python_seed = _python_seed_value(original.random_seeds)
    if python_seed is not None:
        random.seed(python_seed)
    if numpy_seed is not None:
        rng = np.random.default_rng(numpy_seed)
        if isinstance(invocation_seeds, dict):
            invocation_seeds["_numpy_rng"] = rng

    try:
        results = run_fn(dict(original.configuration), invocation_seeds)
    except Exception as exc:  # pragma: no cover - error path still returns artifact.
        recorder.record_failure(exc, context="reproduce_experiment")
        return recorder.finalize()

    recorder.record_result(dict(results))
    return recorder.finalize()


def verify_reproduction(
    original: ExperimentArtifact | str | Path,
    reproduced: ExperimentArtifact | str | Path,
    tolerance: float = 1e-9,
) -> ReproductionCheckResult:
    """Compare two artifacts' results numerically within `tolerance`."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    left = _load_artifact_input(original)
    right = _load_artifact_input(reproduced)
    differing_keys: list[str] = []
    compared_keys = sorted(set(left.results) | set(right.results))
    _compare_values(
        left.results,
        right.results,
        tolerance=tolerance,
        prefix="",
        differing_keys=differing_keys,
    )
    return ReproductionCheckResult(
        matches=not differing_keys,
        tolerance=tolerance,
        differing_keys=differing_keys,
        compared_keys=compared_keys,
    )
