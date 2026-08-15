import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import qes.reproducibility as repro
from qes.reproducibility import (
    ExperimentArtifact,
    ExperimentRecorder,
    ReproductionCheckResult,
    load_artifact,
    reproduce_experiment,
    save_artifact,
    verify_reproduction,
)
from qes.verification import Evidence, VerificationStageResult


def deterministic_run(
    configuration: dict[str, object],
    random_seeds: dict[str, object] | list[object],
) -> dict[str, object]:
    if isinstance(random_seeds, dict):
        rng = random_seeds.get("_numpy_rng")
        if rng is None:
            rng = np.random.default_rng(int(random_seeds["numpy"]))
        python_seed = int(random_seeds["python"])
    else:
        rng = np.random.default_rng(int(random_seeds[0]))
        python_seed = int(random_seeds[1])
    random.seed(python_seed)
    sample_count = int(configuration["draw_count"])
    offset = float(configuration["offset"])
    scale = float(configuration["scale"])
    samples = np.asarray(rng.normal(loc=offset, scale=scale, size=sample_count), dtype=float)
    return {
        "objective": float(samples.sum() + random.random()),
        "samples": samples.tolist(),
        "max_sample": float(samples.max()),
    }


def make_evidence() -> Evidence:
    stage = VerificationStageResult(stage="numerical", passed=True, message="ok")
    return Evidence(candidate="toy", stages=(stage,), passed=True, summary="all checks passed")


def make_full_artifact() -> ExperimentArtifact:
    recorder = ExperimentRecorder(experiment_id="exp-001")
    recorder.record_configuration(draw_count=4, offset=0.25, scale=0.1)
    recorder.record_seed("numpy", 11)
    recorder.record_seed(python=17)
    recorder.record_population({"frontier": np.array([0.2, 0.4, 0.6])})
    recorder.record_lineage_step(step="baseline", parent=None)
    recorder.record_equation({"expression": "mean(samples) + random_bonus"})
    recorder.record_decision("Selected deterministic toy experiment", reason="reproducibility proof")
    recorder.record_resource_allocation(budget=4, iterations=1)
    random.seed(17)
    recorder.record_result(
        deterministic_run(
            {"draw_count": 4, "offset": 0.25, "scale": 0.1},
            {"numpy": 11, "python": 17},
        )
    )
    recorder.record_failure("none observed", severity="info")
    recorder.record_evidence(make_evidence())
    return recorder.finalize()


def test_artifact_to_dict_contains_roadmap_fields():
    artifact = make_full_artifact()
    assert set(artifact.to_dict()) == {
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


def test_artifact_rejects_empty_experiment_id():
    with pytest.raises(ValueError, match="experiment_id"):
        ExperimentArtifact(
            experiment_id="",
            code_version={},
            configuration={},
            random_seeds={},
            hardware={},
        )


def test_artifact_rejects_non_dict_configuration():
    with pytest.raises(TypeError, match="configuration"):
        ExperimentArtifact(
            experiment_id="exp",
            code_version={},
            configuration=[],  # type: ignore[arg-type]
            random_seeds={},
            hardware={},
        )


def test_recorder_records_configuration_merges_kwargs():
    recorder = ExperimentRecorder()
    recorder.record_configuration({"alpha": 1}, beta=2)
    recorder.record_configuration(gamma=3)
    artifact = recorder.finalize()
    assert artifact.configuration == {"alpha": 1, "beta": 2, "gamma": 3}


def test_recorder_record_seed_accepts_name_and_mapping():
    recorder = ExperimentRecorder()
    recorder.record_seed("numpy", 7)
    recorder.record_seed({"python": 13}, extra=5)
    artifact = recorder.finalize()
    assert artifact.random_seeds == {"numpy": 7, "python": 13, "extra": 5}


def test_recorder_record_seed_rejects_value_without_name():
    recorder = ExperimentRecorder()
    with pytest.raises(ValueError, match="requires a seed name"):
        recorder.record_seed(value=3)


def test_recorder_record_population_accepts_numpy_array():
    recorder = ExperimentRecorder()
    recorder.record_population(np.array([1.0, 2.0, 3.0]))
    assert recorder.finalize().population == [1.0, 2.0, 3.0]


def test_recorder_record_population_rejects_invalid_type():
    recorder = ExperimentRecorder()
    with pytest.raises(TypeError, match="population"):
        recorder.record_population("bad")  # type: ignore[arg-type]


def test_recorder_record_lineage_step_requires_content():
    recorder = ExperimentRecorder()
    with pytest.raises(ValueError, match="must not be empty"):
        recorder.record_lineage_step()


def test_recorder_record_equation_appends_entries():
    recorder = ExperimentRecorder()
    recorder.record_equation("x**2")
    recorder.record_equation({"expression": "x + 1"})
    assert recorder.finalize().equations == ["x**2", {"expression": "x + 1"}]


def test_recorder_record_decision_accepts_string_and_mapping():
    recorder = ExperimentRecorder()
    recorder.record_decision("chose narrow search", reason="lower variance")
    recorder.record_decision({"decision": "kept seed", "reason": "deterministic replay"})
    assert recorder.finalize().decisions == [
        {"decision": "chose narrow search", "reason": "lower variance"},
        {"decision": "kept seed", "reason": "deterministic replay"},
    ]


def test_recorder_record_decision_rejects_invalid_type():
    recorder = ExperimentRecorder()
    with pytest.raises(TypeError, match="decision"):
        recorder.record_decision(3)  # type: ignore[arg-type]


def test_recorder_record_resource_allocation_merges():
    recorder = ExperimentRecorder()
    recorder.record_resource_allocation({"budget": 2}, iterations=9)
    recorder.record_resource_allocation(time_seconds=1.5)
    artifact = recorder.finalize()
    assert artifact.resource_allocation == {"budget": 2, "iterations": 9, "time_seconds": 1.5}


def test_recorder_record_result_accepts_key_and_mapping():
    recorder = ExperimentRecorder()
    recorder.record_result("objective", 0.5)
    recorder.record_result({"loss": 0.25}, fitness=0.75)
    artifact = recorder.finalize()
    assert artifact.results == {"objective": 0.5, "loss": 0.25, "fitness": 0.75}


def test_recorder_record_failure_from_exception_and_string():
    recorder = ExperimentRecorder()
    recorder.record_failure(RuntimeError("boom"))
    recorder.record_failure("soft failure", severity="warning")
    artifact = recorder.finalize()
    assert artifact.failures[0]["type"] == "RuntimeError"
    assert artifact.failures[0]["message"] == "boom"
    assert artifact.failures[1] == {"message": "soft failure", "severity": "warning"}


def test_recorder_record_evidence_serializes_evidence():
    recorder = ExperimentRecorder()
    recorder.record_evidence(make_evidence())
    evidence = recorder.finalize().final_evidence
    assert isinstance(evidence, dict)
    assert evidence["evidence_type"] == "qes.verification.Evidence"
    assert evidence["passed"] is True


def test_finalize_produces_complete_artifact():
    artifact = make_full_artifact()
    expected = deterministic_run(
        {"draw_count": 4, "offset": 0.25, "scale": 0.1},
        {"numpy": 11, "python": 17},
    )
    assert artifact.experiment_id == "exp-001"
    assert artifact.code_version["qes_version"]
    assert "git_commit" in artifact.code_version
    assert artifact.hardware["label"] == "local machine info"
    assert artifact.population == {"frontier": [0.2, 0.4, 0.6]}
    assert artifact.lineage == [{"step": "baseline", "parent": None}]
    assert artifact.results["objective"] == pytest.approx(expected["objective"])


def test_finalize_falls_back_when_git_unavailable(monkeypatch: pytest.MonkeyPatch):
    def raise_oserror(*args: object, **kwargs: object) -> object:
        raise OSError("git missing")

    monkeypatch.setattr(repro.subprocess, "run", raise_oserror)
    artifact = ExperimentRecorder().finalize()
    assert artifact.code_version["git_commit"] == "unknown"


def test_context_manager_returns_self_and_records_exception():
    recorder = ExperimentRecorder()
    with pytest.raises(RuntimeError, match="boom"):
        with recorder as active:
            assert active is recorder
            raise RuntimeError("boom")
    artifact = recorder.finalize()
    assert artifact.failures[0]["type"] == "RuntimeError"
    assert artifact.failures[0]["context"] == "context_manager_exit"


def test_save_load_round_trip_with_numpy_fields(tmp_path: pytest.TempPathFactory):
    artifact = ExperimentArtifact(
        experiment_id="exp-round-trip",
        code_version={"qes_version": "0.10.0", "git_commit": "abc123"},
        configuration={"vector": np.array([1, 2, 3])},
        random_seeds={"numpy": 5},
        hardware={
            "label": "local machine info",
            "system": "X",
            "machine": "Y",
            "processor": "Z",
            "python_version": "3.11",
        },
        population={"matrix": np.array([[1.0, 2.0], [3.0, 4.0]])},
        lineage=[{"step": "root"}],
        equations=[np.array([2.0, 3.0])],
        decisions=[{"decision": "keep"}],
        resource_allocation={"budget": 2},
        results={"samples": np.array([0.1, 0.2])},
        failures=[],
        final_evidence={"summary": "ok"},
    )
    path = tmp_path / "artifact.json"
    save_artifact(artifact, path)
    loaded = load_artifact(path)
    assert loaded.configuration["vector"] == [1, 2, 3]
    assert loaded.population["matrix"] == [[1.0, 2.0], [3.0, 4.0]]
    assert loaded.results["samples"] == [0.1, 0.2]


def test_load_artifact_rejects_corrupted_json(tmp_path: pytest.TempPathFactory):
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_artifact(path)


def test_load_artifact_rejects_missing_fields(tmp_path: pytest.TempPathFactory):
    path = tmp_path / "missing.json"
    path.write_text(json.dumps({"experiment_id": "exp"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_artifact(path)


def test_save_artifact_rejects_wrong_type(tmp_path: pytest.TempPathFactory):
    with pytest.raises(TypeError, match="ExperimentArtifact"):
        save_artifact("bad", tmp_path / "artifact.json")  # type: ignore[arg-type]


def test_reproduce_experiment_from_object_recreates_deterministic_results():
    original = make_full_artifact()
    reproduced = reproduce_experiment(original, deterministic_run)
    assert reproduced.experiment_id != original.experiment_id
    assert reproduced.configuration == original.configuration
    assert reproduced.random_seeds == original.random_seeds
    assert reproduced.results == original.results


def test_reproduce_experiment_from_path_recreates_deterministic_results(tmp_path: pytest.TempPathFactory):
    original = make_full_artifact()
    path = tmp_path / "artifact.json"
    save_artifact(original, path)
    reproduced = reproduce_experiment(path, deterministic_run)
    assert reproduced.results == original.results
    assert reproduced.lineage[0]["source_experiment_id"] == original.experiment_id


def test_reproduce_experiment_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        reproduce_experiment(make_full_artifact(), "bad")  # type: ignore[arg-type]


def test_reproduce_experiment_records_failure_when_run_fn_raises():
    def bad_run(
        configuration: dict[str, object],
        random_seeds: dict[str, object] | list[object],
    ) -> dict[str, object]:
        del configuration, random_seeds
        raise RuntimeError("run failed")

    reproduced = reproduce_experiment(make_full_artifact(), bad_run)
    assert reproduced.results == {}
    assert reproduced.failures[0]["type"] == "RuntimeError"


def test_verify_reproduction_reports_match_with_tolerance():
    original = make_full_artifact()
    reproduced = ExperimentArtifact.from_dict(original.to_dict())
    check = verify_reproduction(original, reproduced, tolerance=1e-12)
    assert isinstance(check, ReproductionCheckResult)
    assert check.matches is True
    assert check.differing_keys == []
    assert "objective" in check.compared_keys


def test_verify_reproduction_reports_mismatch_and_missing_keys():
    original = make_full_artifact()
    reproduced = ExperimentArtifact.from_dict(original.to_dict())
    reproduced.results["objective"] = reproduced.results["objective"] + 1.0
    del reproduced.results["max_sample"]
    check = verify_reproduction(original, reproduced, tolerance=1e-12)
    assert check.matches is False
    assert "objective" in check.differing_keys
    assert "max_sample" in check.differing_keys


def test_verify_reproduction_supports_nested_lists():
    original = make_full_artifact()
    reproduced = ExperimentArtifact.from_dict(original.to_dict())
    reproduced.results["samples"][0] += 1e-12
    check = verify_reproduction(original, reproduced, tolerance=1e-9)
    assert check.matches is True


def test_verify_reproduction_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="non-negative"):
        verify_reproduction(make_full_artifact(), make_full_artifact(), tolerance=-1.0)


@dataclass
class _MiniDataclass:
    value: int


class _ToDictObject:
    def to_dict(self) -> dict[str, object]:
        return {"value": Path("demo.txt")}


def test_json_ready_and_mapping_helpers_cover_additional_types():
    result = repro._json_ready(
        {
            "numpy_scalar": np.int64(3),
            "path": Path("artifact.json"),
            "error": RuntimeError("boom"),
            "dataclass": _MiniDataclass(4),
            "to_dict": _ToDictObject(),
            "set_values": {1, 2},
            "fallback": object(),
        }
    )
    assert result["numpy_scalar"] == 3
    assert result["path"] == "artifact.json"
    assert result["error"] == {"type": "RuntimeError", "message": "boom"}
    assert result["dataclass"] == {"value": 4}
    assert result["to_dict"] == {"value": "demo.txt"}
    assert sorted(result["set_values"]) == [1, 2]
    assert isinstance(result["fallback"], str)

    with pytest.raises(TypeError, match="mapping"):
        repro._normalized_mapping([])  # type: ignore[arg-type]


def test_capture_helpers_and_seed_extractors_cover_fallback_paths(monkeypatch: pytest.MonkeyPatch):
    class _Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(repro.subprocess, "run", lambda *args, **kwargs: _Result())
    assert repro._capture_git_commit() == "unknown"
    assert repro._capture_hardware()["label"] == "local machine info"

    assert repro._seed_value({"numpy": True, "other": 7}) == 7
    assert repro._seed_value(["bad", 5.2]) == 5
    assert repro._seed_value(["bad", False]) is None
    assert repro._seed_value({"name": "seedless"}) is None
    assert repro._python_seed_value({"python": 9.8}) == 9
    assert repro._python_seed_value({"seed": 4}) == 4


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("code_version", [], "code_version"),
        ("random_seeds", "bad", "random_seeds"),
        ("hardware", [], "hardware"),
        ("population", "bad", "population"),
        ("lineage", "bad", "lineage"),
        ("equations", {}, "equations"),
        ("decisions", {}, "decisions"),
        ("resource_allocation", [], "resource_allocation"),
        ("results", [], "results"),
        ("failures", {}, "failures"),
        ("final_evidence", "bad", "final_evidence"),
    ],
)
def test_artifact_validates_additional_field_types(field: str, value: object, match: str):
    kwargs = dict(
        experiment_id="exp",
        code_version={},
        configuration={},
        random_seeds={},
        hardware={},
    )
    kwargs[field] = value
    with pytest.raises(TypeError, match=match):
        ExperimentArtifact(**kwargs)  # type: ignore[arg-type]


def test_reproduction_check_result_strings_and_recorder_exit_without_exception():
    assert str(ReproductionCheckResult(True, 1e-3, [], ["objective"])) == (
        "match within tolerance=0.001 for keys=['objective']"
    )
    mismatch = ReproductionCheckResult(False, 1e-6, ["objective"], ["objective"])
    assert "mismatch" in str(mismatch)

    recorder = ExperimentRecorder()
    with recorder:
        recorder.record_configuration(alpha=1)
    assert recorder.finalize().failures == []


def test_recorder_validation_and_knowledge_graph_lineage():
    recorder = ExperimentRecorder()
    with pytest.raises(ValueError, match="seed name"):
        recorder.record_seed("", 1)
    with pytest.raises(ValueError, match="result key"):
        recorder.record_result("", 1)
    with pytest.raises(TypeError, match="result"):
        recorder.record_result(1)  # type: ignore[arg-type]
    recorder.record_failure({"message": "mapped"})
    with pytest.raises(TypeError, match="failure"):
        recorder.record_failure(1)  # type: ignore[arg-type]

    assert recorder.finalize().failures == [{"message": "mapped"}]

    recorder = ExperimentRecorder()
    assert recorder._knowledge_graph is not None
    recorder._knowledge_graph.add_node("result", {"value": 1})
    recorder.record_lineage_step(step="baseline")
    artifact = recorder.finalize()
    assert isinstance(artifact.lineage, dict)
    assert artifact.lineage["entries"] == [{"step": "baseline"}]
    assert "knowledge_graph" in artifact.lineage


def test_load_artifact_and_reproduce_cover_file_and_list_seed_paths(tmp_path: pytest.TempPathFactory):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_artifact(missing_path)

    non_object_path = tmp_path / "array.json"
    non_object_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_artifact(non_object_path)

    artifact = ExperimentArtifact(
        experiment_id="exp-list",
        code_version={"qes_version": "0.1", "git_commit": "abc"},
        configuration={"draw_count": 2, "offset": 0.0, "scale": 1.0},
        random_seeds=[5, 9],
        hardware={
            "label": "local machine info",
            "system": "X",
            "machine": "Y",
            "processor": "Z",
            "python_version": "3.11",
        },
    )

    reproduced = reproduce_experiment(artifact, deterministic_run)
    assert reproduced.results["samples"]
    assert reproduced.random_seeds == {}

    seedless = ExperimentArtifact(
        experiment_id="exp-seedless",
        code_version={"qes_version": "0.1", "git_commit": "abc"},
        configuration={"draw_count": 1, "offset": 0.0, "scale": 1.0},
        random_seeds={"label": "none"},
        hardware=artifact.hardware,
    )
    replayed = reproduce_experiment(seedless, lambda configuration, random_seeds: {"echo": [configuration, random_seeds]})
    assert replayed.results["echo"][1] == {"label": "none"}


def test_verify_reproduction_reports_list_length_and_scalar_mismatches():
    original = ExperimentArtifact(
        experiment_id="exp-a",
        code_version={},
        configuration={},
        random_seeds={},
        hardware={},
        results={"labels": ["a"], "mode": "x"},
    )
    reproduced = ExperimentArtifact(
        experiment_id="exp-b",
        code_version={},
        configuration={},
        random_seeds={},
        hardware={},
        results={"labels": ["a", "b"], "mode": "y"},
    )
    check = verify_reproduction(original, reproduced)
    assert check.matches is False
    assert "labels" in check.differing_keys
    assert "mode" in check.differing_keys
