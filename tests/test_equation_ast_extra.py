import numpy as np

from qes.equation_ast import ASTNode, Constant, EquationASTForge, Operator


def test_simplify_invalid_input_raises_typeerror():
    forge = EquationASTForge(population_size=4, rng=np.random.default_rng(0))
    import pytest

    with pytest.raises(TypeError):
        forge.simplify("not-an-ast")


class _TrickyNode(ASTNode):
    """Returns finite for exact sample, but non-finite for perturbed samples."""

    def evaluate(self, x: np.ndarray) -> float:
        # If any element differs from 0.0 by more than 1e-9, return inf
        arr = np.asarray(x, dtype=float)
        if np.any(np.abs(arr - 0.0) > 1e-9):
            return float(np.inf)
        return 0.0

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1


def test_analyze_stability_shifted_not_finite():
    forge = EquationASTForge(population_size=4, rng=np.random.default_rng(1))
    node = _TrickyNode()
    states = np.array([[0.0]])
    report = forge.analyze_stability(node, states, epsilon=1e-3, threshold=1.0)
    assert not report.passed
    assert report.max_sensitivity == float("inf") or report.failing_dimension == 0


def test_initial_population_includes_neg_and_mul():
    # Create a small population size so we deterministically hit the neg/* additions
    forge = EquationASTForge(population_size=6, rng=np.random.default_rng(2))
    pop = forge._initial_population(dim=1, variable_names=None, seeds=None)
    # ensure at least one Operator exists (neg or *)
    assert any(isinstance(e.root, Operator) and e.root.name in {"neg", "*"} for e in pop)


def test_simplify_with_ast_node_returns_simplified():
    forge = EquationASTForge(population_size=4, rng=np.random.default_rng(0))
    node = Constant(2.0)
    simplified = forge.simplify(node)
    assert isinstance(simplified, Constant)
    assert simplified.value == 2.0
