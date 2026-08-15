from __future__ import annotations

import builtins
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest

import qes.observatory as observatory
from qes.observatory import (
    ObservatoryDashboard,
    ObservatorySnapshot,
    build_snapshot_from_modules,
    render_reality_tree,
)

try:
    from qes.digital_twin_loop import AnomalyReport, DriftReport
except ImportError:  # pragma: no cover - defensive against concurrent workspace state
    AnomalyReport = DriftReport = None

try:
    from qes.divergence import DSA
except ImportError:  # pragma: no cover - defensive against concurrent workspace state
    DSA = None

try:
    from qes.knowledge_graph import KnowledgeGraph
except ImportError:  # pragma: no cover - defensive against concurrent workspace state
    KnowledgeGraph = None

try:
    from qes.marketplace import MarketBid, RealityAccount, RealityMarketplace
except ImportError:  # pragma: no cover - defensive against concurrent workspace state
    MarketBid = RealityAccount = RealityMarketplace = None

try:
    from qes.permission import GenesisPermission
except ImportError:  # pragma: no cover - defensive against concurrent workspace state
    GenesisPermission = None


def make_snapshot(**overrides: object) -> ObservatorySnapshot:
    payload = {
        "population": 12_482,
        "active_realities": 4_281,
        "compute_pct": 83.4,
        "convergence": 0.73,
        "risk": 0.08,
        "novelty": 0.41,
        "reality_tree": {
            "ROOT": ["R1", "R2", "R3"],
            "R1": ["R4"],
            "R2": ["R5", "R6"],
            "R3": ["R7"],
            "R4": [],
            "R5": [],
            "R6": [],
            "R7": [],
        },
    }
    payload.update(overrides)
    return ObservatorySnapshot(**payload)


def test_snapshot_accepts_required_core_fields_only() -> None:
    snapshot = make_snapshot()

    assert snapshot.population == 12_482
    assert snapshot.equations is None
    assert snapshot.discovered_patterns is None


def test_snapshot_accepts_full_optional_payload() -> None:
    snapshot = make_snapshot(
        equations=[{"expr": "x**2 - 0.4*x"}],
        divergence={"dr": 0.12},
        permission={"admitted": True},
        resource_allocation={"R1": {"compute": 10.0}},
        agent_interactions=[{"agent": "safety"}],
        digital_twin_error={"max_z_score": 1.5},
        failures={"overall_failure_rate": 0.0},
        discovered_patterns=[{"pattern": "exploit-local-minimum"}],
    )

    assert snapshot.equations == [{"expr": "x**2 - 0.4*x"}]
    assert snapshot.failures == {"overall_failure_rate": 0.0}


def test_snapshot_rejects_invalid_compute_percentage() -> None:
    with pytest.raises(ValueError, match="compute_pct must be <= 100"):
        make_snapshot(compute_pct=120.0)


def test_snapshot_rejects_negative_population() -> None:
    with pytest.raises(ValueError, match="population must be >= 0"):
        make_snapshot(population=-1)


def test_render_contains_all_core_metrics_with_formatting() -> None:
    rendered = ObservatoryDashboard(make_snapshot()).render()

    assert "Population        12,482" in rendered
    assert "Active Realities  4,281" in rendered
    assert "Compute           83.4%" in rendered
    assert "Convergence       0.73" in rendered
    assert "Risk              0.08" in rendered
    assert "Novelty           0.41" in rendered


def test_render_includes_optional_sections_only_when_present() -> None:
    snapshot = make_snapshot(
        divergence={"state": "stable"},
        permission={"admitted": True},
        digital_twin_error={"flagged": False},
    )

    rendered = ObservatoryDashboard(snapshot).render()

    assert "Divergence:" in rendered
    assert "Permission:" in rendered
    assert "Digital Twin Error:" in rendered
    assert "Failures:" not in rendered


def test_render_live_streams_each_snapshot() -> None:
    outputs: list[str] = []
    snapshots = [make_snapshot(population=1), make_snapshot(population=2)]

    ObservatoryDashboard.render_live(snapshots, refresh_fn=outputs.append)

    assert len(outputs) == 2
    assert "Population        1" in outputs[0]
    assert "Population        2" in outputs[1]


def test_render_live_rejects_non_callable_refresh_function() -> None:
    with pytest.raises(TypeError, match="refresh_fn must be callable"):
        ObservatoryDashboard.render_live([make_snapshot()], refresh_fn=123)


def test_dashboard_update_rejects_wrong_type() -> None:
    dashboard = ObservatoryDashboard()

    with pytest.raises(TypeError, match="snapshot must be an ObservatorySnapshot"):
        dashboard.update("not-a-snapshot")  # type: ignore[arg-type]


def test_dashboard_render_requires_snapshot() -> None:
    with pytest.raises(ValueError, match="no ObservatorySnapshot has been loaded"):
        ObservatoryDashboard().render()


def test_render_reality_tree_single_node() -> None:
    rendered = render_reality_tree({"ROOT": []}, "ROOT")

    assert rendered == "ROOT"


def test_render_reality_tree_linear_chain() -> None:
    tree = {"ROOT": ["R1"], "R1": ["R2"], "R2": ["R3"], "R3": []}

    rendered = render_reality_tree(tree, "ROOT")

    assert rendered.splitlines() == ["ROOT", "└── R1", "    └── R2", "        └── R3"]


def test_render_reality_tree_branching_tree() -> None:
    tree = {"ROOT": ["A", "B"], "A": ["A1"], "B": ["B1", "B2"], "A1": [], "B1": [], "B2": []}

    rendered = render_reality_tree(tree, "ROOT")

    assert "├── A" in rendered
    assert "│   └── A1" in rendered
    assert "└── B" in rendered
    assert "    ├── B1" in rendered
    assert "    └── B2" in rendered


def test_render_reality_tree_matches_roadmap_shape() -> None:
    tree = {
        "ROOT": ["R1", "R2", "R3"],
        "R1": ["R4"],
        "R2": ["R5", "R6"],
        "R3": ["R7"],
        "R4": [],
        "R5": [],
        "R6": [],
        "R7": [],
    }

    rendered = render_reality_tree(tree, "ROOT")

    assert rendered.splitlines() == [
        "ROOT",
        "├── R1",
        "│   └── R4",
        "├── R2",
        "│   ├── R5",
        "│   └── R6",
        "└── R3",
        "    └── R7",
    ]


def test_render_reality_tree_rejects_empty_tree() -> None:
    with pytest.raises(ValueError, match="reality_tree must not be empty"):
        render_reality_tree({}, "ROOT")


def test_render_reality_tree_rejects_missing_root_key() -> None:
    with pytest.raises(ValueError, match="root 'ROOT' is not present"):
        render_reality_tree({"R1": []}, "ROOT")


def test_render_reality_tree_rejects_malformed_adjacency() -> None:
    with pytest.raises(TypeError, match="sequence of child node names"):
        render_reality_tree({"ROOT": "R1"}, "ROOT")  # type: ignore[arg-type]


def test_render_reality_tree_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="cycle detected"):
        render_reality_tree({"ROOT": ["R1"], "R1": ["ROOT"]}, "ROOT")


def test_build_snapshot_falls_back_without_optional_modules() -> None:
    snapshot = build_snapshot_from_modules(population=3, reality_tree={"ROOT": []})

    assert snapshot.population == 3
    assert snapshot.equations is None
    assert snapshot.discovered_patterns is None


@pytest.mark.skipif(DSA is None or GenesisPermission is None, reason="optional integrations unavailable")
def test_build_snapshot_assembles_real_divergence_and_permission_objects() -> None:
    divergence = DSA().update(
        x=np.array([1.0, 2.0]),
        x_star=np.zeros(2),
        dt=1.0,
        w=np.eye(2),
    )
    permission = GenesisPermission(theta=1.0).evaluate(
        x=np.zeros(2),
        lower=-np.ones(2),
        upper=np.ones(2),
        w=np.ones(2),
    )

    snapshot = build_snapshot_from_modules(
        population=9,
        active_realities=2,
        compute_pct=55.0,
        convergence=0.4,
        risk=0.2,
        novelty=0.3,
        reality_tree={"ROOT": []},
        divergence=divergence,
        permission=permission,
    )

    assert snapshot.divergence == {
        "d": [1.0, 2.0],
        "d_dot": [0.0, 0.0],
        "d_ddot": [0.0, 0.0],
        "dr": 5.0,
        "health": 5.0,
    }
    assert snapshot.permission == {
        "phi": 0.0,
        "cci": 0.0,
        "margin": 0.5,
        "hard_permission": True,
        "soft_permission": 1.0,
        "admitted": True,
    }


@pytest.mark.skipif(KnowledgeGraph is None, reason="knowledge graph unavailable")
def test_build_snapshot_derives_equations_and_patterns_from_knowledge_graph() -> None:
    graph = KnowledgeGraph()
    reality = graph.add_node("reality", payload={"seed": 7})
    observation = graph.add_node("observation", payload={"note": "steady"}, parents=[reality.id])
    equation = graph.add_node("equation", payload={"expr": "x**2"}, parents=[observation.id])
    pattern = graph.add_node("pattern", payload={"bias": "exploit"}, parents=[equation.id], success=True)
    result = graph.record_experiment(equation.id, {"fitness": 0.9}, success=True, pattern_id=pattern.id)
    graph.record_causal_hypothesis(equation.id, result.id, confidence=0.8)

    snapshot = build_snapshot_from_modules(
        population=5,
        reality_tree={"ROOT": []},
        knowledge_graph=graph,
    )

    assert snapshot.equations == [{"id": equation.id, "payload": {"expr": "x**2"}}]
    assert snapshot.discovered_patterns == {
        "patterns": [{"id": pattern.id, "payload": {"bias": "exploit"}, "success": True}],
        "causal_hypotheses": [
            {
                "cause_id": equation.id,
                "effect_id": result.id,
                "confidence": 0.8,
                "evidence_count": 1,
            }
        ],
    }


@pytest.mark.skipif(
    RealityMarketplace is None or RealityAccount is None or MarketBid is None,
    reason="marketplace unavailable",
)
def test_build_snapshot_summarizes_marketplace_allocations() -> None:
    marketplace = RealityMarketplace()
    marketplace.register(
        RealityAccount("alpha", compute_budget=10.0, memory_budget=5.0, energy_budget=2.0)
    )
    marketplace.submit_bid(
        MarketBid("alpha", requested_resources={"compute": 4.0}, uncertainty=0.2)
    )
    marketplace.clear_market({"compute": 4.0})

    snapshot = build_snapshot_from_modules(
        population=1,
        reality_tree={"ROOT": []},
        resource_allocation=marketplace,
    )

    assert snapshot.resource_allocation == {
        "round_index": 1,
        "accounts": ["alpha"],
        "last_allocations": {
            "alpha": {"compute": 4.0, "memory": 0.0, "energy": 0.0}
        },
    }


@pytest.mark.skipif(AnomalyReport is None or DriftReport is None, reason="digital twin loop unavailable")
def test_build_snapshot_accepts_real_digital_twin_reports() -> None:
    anomaly = AnomalyReport(
        flagged=False,
        residual=np.array([0.1, -0.1]),
        z_scores=np.array([0.5, 0.5]),
        threshold=3.0,
        max_z_score=0.5,
    )
    drift = DriftReport(
        flagged=False,
        baseline_mean=0.1,
        baseline_std=0.02,
        recent_mean=0.12,
        recent_std=0.03,
        mean_shift_sigma=1.0,
        std_ratio=1.5,
    )

    snapshot = build_snapshot_from_modules(
        population=1,
        reality_tree={"ROOT": []},
        digital_twin_error={"anomaly": anomaly, "drift": drift},
    )

    assert snapshot.digital_twin_error == {
        "anomaly": {
            "flagged": False,
            "residual": [0.1, -0.1],
            "z_scores": [0.5, 0.5],
            "threshold": 3.0,
            "max_z_score": 0.5,
        },
        "drift": {
            "flagged": False,
            "baseline_mean": 0.1,
            "baseline_std": 0.02,
            "recent_mean": 0.12,
            "recent_std": 0.03,
            "mean_shift_sigma": 1.0,
            "std_ratio": 1.5,
        },
    }


@dataclass
class DummyPayload:
    score: float
    label: str


def test_build_snapshot_converts_plain_dataclass_payloads() -> None:
    snapshot = build_snapshot_from_modules(
        population=1,
        reality_tree={"ROOT": []},
        agent_interactions=DummyPayload(score=0.9, label="coalition"),
    )

    assert snapshot.agent_interactions == {"score": 0.9, "label": "coalition"}


def test_private_validators_reject_bad_scalar_and_tree_inputs() -> None:
    with pytest.raises(TypeError, match="population must be an integer"):
        observatory._validate_non_negative_int("population", True)
    with pytest.raises(TypeError, match="compute_pct must be a real number"):
        observatory._validate_finite_float("compute_pct", True)
    with pytest.raises(TypeError, match="compute_pct must be a real number"):
        observatory._validate_finite_float("compute_pct", object())
    with pytest.raises(ValueError, match="compute_pct must be finite"):
        observatory._validate_finite_float("compute_pct", float("inf"))
    with pytest.raises(ValueError, match="compute_pct must be >= 0.0"):
        observatory._validate_finite_float("compute_pct", -1.0, minimum=0.0)
    with pytest.raises(TypeError, match="reality_tree must be a mapping"):
        observatory._validate_reality_tree([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reality_tree keys must be non-empty strings"):
        observatory._validate_reality_tree({"": []})
    with pytest.raises(ValueError, match="contain only non-empty string child names"):
        observatory._validate_reality_tree({"ROOT": [""]})


def test_private_formatting_helpers_cover_scalar_collection_and_nested_paths() -> None:
    class Flag(Enum):
        READY = "ready"

    assert observatory._infer_root({"A": ["B"], "B": ["A"]}) == "A"
    assert observatory._to_plain_data(None) is None
    assert observatory._to_plain_data(np.array([1, 2])) == [1, 2]
    assert observatory._to_plain_data(np.float64(1.5)) == pytest.approx(1.5)
    assert observatory._to_plain_data(Flag.READY) == "ready"
    assert observatory._format_scalar(1234) == "1,234"
    assert observatory._format_scalar(1.25) == "1.25"
    assert observatory._format_block(None) == []
    assert observatory._format_block({"outer": {"inner": 1}}) == [
        "  outer:",
        "    inner: 1",
    ]
    assert observatory._format_block([1, {"k": 2}, [3]]) == [
        "  - 1",
        "  -",
        "    k: 2",
        "  -",
        "    - 3",
    ]
    assert observatory._format_block("done") == ["  done"]


def test_extract_knowledge_graph_sections_handles_fallback_and_error_paths() -> None:
    class NoNodes:
        pass

    assert observatory._extract_knowledge_graph_sections(NoNodes()) == (None, None)

    class BrokenNodes:
        def nodes(self, kind: str | None = None):
            if kind == "equation":
                raise RuntimeError("eq")
            if kind == "pattern":
                raise RuntimeError("pattern")
            return [type("Node", (), {"id": 1})()]

        def causal_hypotheses_for(self, node_id: str):
            return []

    assert observatory._extract_knowledge_graph_sections(BrokenNodes()) == (None, None)

    class HypothesisOnly:
        def nodes(self, kind: str | None = None):
            if kind == "equation":
                return []
            if kind == "pattern":
                return []
            return [
                type("Node", (), {"id": None})(),
                type("Node", (), {"id": "n-1"})(),
            ]

        def causal_hypotheses_for(self, node_id: str):
            if node_id == "n-1":
                return [
                    type(
                        "Hypothesis",
                        (),
                        {
                            "cause_id": "c-1",
                            "effect_id": "e-1",
                            "confidence": 0.8,
                            "evidence_count": 2,
                        },
                    )()
                ]
            return []

    equations, patterns = observatory._extract_knowledge_graph_sections(HypothesisOnly())
    assert equations is None
    assert patterns == {"causal_hypotheses": [{"cause_id": "c-1", "effect_id": "e-1", "confidence": 0.8, "evidence_count": 2}]}

    class BrokenHypotheses:
        def nodes(self, kind: str | None = None):
            if kind == "equation":
                return []
            if kind == "pattern":
                return []
            return [type("Node", (), {"id": "n-1"})()]

        def causal_hypotheses_for(self, node_id: str):
            return [
                type(
                    "Hypothesis",
                    (),
                    {
                        "cause_id": "c-1",
                        "effect_id": "e-1",
                        "confidence": "bad",
                        "evidence_count": 2,
                    },
                )()
            ]

    assert observatory._extract_knowledge_graph_sections(BrokenHypotheses()) == (None, None)


def test_extract_knowledge_graph_sections_handles_non_dict_merge_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_isinstance = builtins.isinstance

    def fake_isinstance(obj: object, cls: object) -> bool:
        if cls is dict and obj == {}:
            return False
        return real_isinstance(obj, cls)

    monkeypatch.setattr(builtins, "isinstance", fake_isinstance)

    class HypothesisOnly:
        def nodes(self, kind: str | None = None):
            if kind == "equation":
                return []
            if kind == "pattern":
                return []
            return [type("Node", (), {"id": "n-1"})()]

        def causal_hypotheses_for(self, node_id: str):
            return [
                type(
                    "Hypothesis",
                    (),
                    {
                        "cause_id": "c-1",
                        "effect_id": "e-1",
                        "confidence": 0.8,
                        "evidence_count": 2,
                    },
                )()
            ]

    assert observatory._extract_knowledge_graph_sections(HypothesisOnly()) == (None, {})


def test_summarize_marketplace_and_dashboard_render_skip_empty_blocks() -> None:
    class ThinMarketplace:
        round_index = 3

        def get_last_allocations(self):
            return {"alpha": {"compute": 1.0}}

    @dataclass
    class BrokenThinMarketplace:
        round_index: int = 7

        def get_last_allocations(self):
            raise RuntimeError("boom")

    assert observatory._summarize_marketplace(ThinMarketplace()) == {
        "round_index": 3,
        "last_allocations": {"alpha": {"compute": 1.0}},
    }
    assert observatory._summarize_marketplace(BrokenThinMarketplace()) == {"round_index": 7}
    assert observatory._summarize_marketplace({"round_index": 9}) == {"round_index": 9}

    rendered = ObservatoryDashboard(
        make_snapshot(
            equations=[],
            divergence={},
            permission=[],
        )
    ).render()
    assert "Equations:" not in rendered
    assert "Permission:" not in rendered


def test_render_reality_tree_handles_leaf_children_not_present_as_keys() -> None:
    rendered = render_reality_tree({"ROOT": ["leaf"]}, "ROOT")
    assert rendered.splitlines() == ["ROOT", "└── leaf"]


def test_build_snapshot_preserves_explicit_equations_and_patterns_over_graph_extraction() -> None:
    class Graph:
        def nodes(self, kind: str | None = None):
            if kind == "equation":
                return [type("Node", (), {"id": "eq", "payload": {"expr": "x"}})()]
            if kind == "pattern":
                return [type("Node", (), {"id": "pat", "payload": {"note": "graph"}, "success": True})()]
            return []

        def causal_hypotheses_for(self, node_id: str):
            return []

    snapshot = build_snapshot_from_modules(
        population=1,
        reality_tree={"ROOT": []},
        equations=[{"expr": "explicit"}],
        discovered_patterns={"patterns": [{"id": "explicit"}]},
        knowledge_graph=Graph(),
    )

    assert snapshot.equations == [{"expr": "explicit"}]
    assert snapshot.discovered_patterns == {"patterns": [{"id": "explicit"}]}
