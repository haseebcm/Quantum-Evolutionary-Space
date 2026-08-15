import numpy as np

from qes.reproducibility import ExperimentArtifact, verify_reproduction
from qes.sdk import QESClient


def test_qesclient_candidate_and_best_snapshot():
    bounds = ([0.0], [1.0])

    # provide a trivial step_fn so the client builds without running
    client = QESClient(bounds, step_fn=lambda room: None, population=2)
    state, score = client._candidate_for_room(client.seed_room)
    assert isinstance(state, np.ndarray)
    assert score is None

    # best snapshot should return a dominant room id and either a score or None
    best_id, best_state, best_score, weight = client._best_snapshot()
    assert best_id is None or isinstance(best_id, str)


def test_verify_reproduction_detects_differences():
    a = ExperimentArtifact(
        experiment_id="e1",
        code_version={"qes_version": "1.0", "git_commit": "abc"},
        configuration={},
        random_seeds={},
        hardware={},
        population={},
        lineage=[],
        equations=[],
        decisions=[],
        resource_allocation={},
        results={"value": 1.0},
        failures=[],
        final_evidence=None,
    )
    b = ExperimentArtifact(
        experiment_id="e2",
        code_version={"qes_version": "1.0", "git_commit": "abc"},
        configuration={},
        random_seeds={},
        hardware={},
        population={},
        lineage=[],
        equations=[],
        decisions=[],
        resource_allocation={},
        results={"value": 2.0},
        failures=[],
        final_evidence=None,
    )
    result = verify_reproduction(a, b, tolerance=1e-9)
    assert not result.matches
    assert "value" in result.differing_keys
