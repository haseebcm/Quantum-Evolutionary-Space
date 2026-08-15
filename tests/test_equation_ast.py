import numpy as np
import pytest

import qes.equation_ast as equation_ast_module
from qes.equation_ast import (
    ASTNode,
    Constant,
    EquationAST,
    EquationASTForge,
    Operator,
    Variable,
    iter_subtree_paths,
    replace_subtree,
    simplify_ast,
    subtree_at,
)


def test_variable_can_be_constructed_from_name():
    variable = Variable.from_name("velocity", ["position", "velocity"])
    assert variable.index == 1
    assert variable.name == "velocity"


def test_ast_evaluation_correct_for_each_primitive():
    x = np.array([4.0, 0.5])
    cases: list[tuple[ASTNode, float]] = [
        (Constant(3.0), 3.0),
        (Variable(index=0, name="x0"), 4.0),
        (Operator("+", (Constant(1.0), Constant(2.0))), 3.0),
        (Operator("-", (Constant(5.0), Constant(2.0))), 3.0),
        (Operator("*", (Constant(3.0), Constant(2.0))), 6.0),
        (Operator("/", (Constant(9.0), Constant(3.0))), 3.0),
        (Operator("safe_div", (Constant(9.0), Constant(0.0))), 0.0),
        (Operator("pow", (Constant(3.0), Constant(2.0))), 9.0),
        (Operator("safe_pow", (Constant(-4.0), Constant(0.5))), -2.0),
        (Operator("exp", (Constant(1.0),)), float(np.exp(1.0))),
        (Operator("sin", (Constant(np.pi / 2.0),)), 1.0),
        (Operator("cos", (Constant(0.0),)), 1.0),
        (Operator("log", (Constant(np.e),)), 1.0),
        (Operator("safe_log", (Constant(0.0),)), float(np.log(1e-12))),
        (Operator("sqrt", (Constant(9.0),)), 3.0),
        (Operator("safe_sqrt", (Constant(-1.0),)), 0.0),
        (Operator("neg", (Constant(7.0),)), -7.0),
    ]

    for node, expected in cases:
        assert np.isclose(node.evaluate(x), expected)


def test_depth_node_count_and_subtree_replacement_work():
    tree = Operator(
        "*",
        (
            Operator("+", (Variable(0), Constant(1.0))),
            Operator("neg", (Variable(1),)),
        ),
    )

    assert tree.depth() == 3
    assert tree.node_count() == 6

    paths = iter_subtree_paths(tree)
    assert () in paths
    replaced = replace_subtree(tree, (0, 1), Constant(3.0))
    assert subtree_at(replaced, (0, 1)) == Constant(3.0)
    assert np.isclose(replaced.evaluate(np.array([2.0, 4.0])), -20.0)


def test_mutation_produces_valid_evaluable_tree():
    forge = EquationASTForge(rng=np.random.default_rng(2), max_depth=4)
    base = EquationAST(
        root=Operator(
            "+",
            (
                Variable(0),
                Operator("*", (Variable(1), Constant(2.0))),
            ),
        )
    )

    mutated = forge.mutate(base, dim=2)
    sample = np.array([[0.5, 1.5], [1.0, 2.0]])

    assert mutated.parent == base.id
    assert mutated.generation == base.generation + 1
    assert mutated.depth() <= forge.max_depth
    assert forge.validate_dimensions(mutated.root, dim=2).passed
    assert forge.validate_numerically(mutated.root, sample).passed


def test_crossover_produces_valid_tree():
    forge = EquationASTForge(rng=np.random.default_rng(7), max_depth=4)
    parent_a = EquationAST(root=Operator("+", (Variable(0), Constant(1.0))))
    parent_b = EquationAST(root=Operator("*", (Variable(1), Variable(1))))

    child = forge.crossover(parent_a, parent_b)
    sample = np.array([[1.5, 2.0], [0.25, 0.5]])

    assert child.parent == parent_a.id
    assert child.co_parent == parent_b.id
    assert child.generation == 1
    assert child.depth() <= forge.max_depth
    assert forge.validate_dimensions(child.root, dim=2).passed
    assert forge.validate_numerically(child.root, sample).passed


def test_simplification_folds_constants_and_removes_no_ops():
    folded = simplify_ast(Operator("+", (Constant(1.0), Constant(2.0))))
    assert folded == Constant(3.0)

    redundant = Operator("*", (Operator("+", (Variable(0), Constant(0.0))), Constant(1.0)))
    simplified = simplify_ast(redundant)
    assert simplified == Variable(0)
    assert simplified.node_count() < redundant.node_count()


def test_dimensional_validation_catches_out_of_range_variable():
    forge = EquationASTForge(rng=np.random.default_rng(0))
    report = forge.validate_dimensions(Operator("+", (Variable(0), Variable(2))), dim=2)
    assert not report.passed
    assert report.invalid_indices == (2,)


def test_numerical_validation_catches_non_finite_outputs():
    forge = EquationASTForge(rng=np.random.default_rng(0))
    node = Operator("/", (Constant(1.0), Variable(0)))
    report = forge.validate_numerically(node, np.array([[0.0], [1.0]]))
    assert not report.passed
    assert report.failing_sample == 0


def test_stability_analysis_flags_unstable_expression_but_accepts_stable_one():
    forge = EquationASTForge(rng=np.random.default_rng(0))
    unstable = Operator("/", (Constant(1.0), Variable(0)))
    stable = Variable(0)
    probe_states = np.array([[1e-4], [2e-4], [5e-4]])

    unstable_report = forge.analyze_stability(unstable, probe_states, threshold=1e4)
    stable_report = forge.analyze_stability(stable, probe_states, threshold=10.0)

    assert not unstable_report.passed
    assert unstable_report.max_sensitivity > unstable_report.threshold
    assert stable_report.passed
    assert stable_report.max_sensitivity <= stable_report.threshold


def test_end_to_end_evolution_pipeline_improves_fitness():
    rng = np.random.default_rng(21)
    forge = EquationASTForge(
        population_size=8,
        elite_fraction=0.25,
        mutation_rate=0.95,
        crossover_rate=0.7,
        max_depth=4,
        rng=rng,
    )
    samples = np.linspace(-1.0, 1.0, 41, dtype=float).reshape(-1, 1)
    targets = samples[:, 0] ** 2

    def score_fn(node: ASTNode) -> float:
        predictions = np.asarray([node.evaluate(sample) for sample in samples], dtype=float)
        return float(np.mean((predictions - targets) ** 2))

    seeds = [
        Constant(0.0),
        Constant(1.0),
        Constant(-1.0),
        Variable(0),
        Operator("neg", (Variable(0),)),
        Operator("+", (Variable(0), Constant(1.0))),
        Operator("-", (Constant(1.0), Variable(0))),
        Operator("*", (Variable(0), Constant(0.5))),
    ]

    result = forge.evolve(
        dim=1,
        score_fn=score_fn,
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
        generations=14,
        seeds=seeds,
        sample_count=64,
        stability_threshold=200.0,
        complexity_penalty=1e-4,
    )

    assert len(result.history) >= 2
    assert result.history[0].best_objective > result.history[-1].best_objective
    assert result.best_equation.objective < 0.08


def test_primitives_and_nodes_validate_inputs_and_string_forms():
    with pytest.raises(ValueError, match="one-dimensional"):
        Constant(1.0).evaluate(np.zeros((1, 1)))

    with pytest.raises(TypeError, match="name must be a string"):
        equation_ast_module.Primitive(name=1, arity=1, fn=lambda values: 0.0)
    with pytest.raises(ValueError, match="name must be non-empty"):
        equation_ast_module.Primitive(name="", arity=1, fn=lambda values: 0.0)
    with pytest.raises(TypeError, match="arity must be an integer"):
        equation_ast_module.Primitive(name="x", arity=1.5, fn=lambda values: 0.0)
    with pytest.raises(ValueError, match="arity must be >= 1"):
        equation_ast_module.Primitive(name="x", arity=0, fn=lambda values: 0.0)
    with pytest.raises(TypeError, match="fn must be callable"):
        equation_ast_module.Primitive(name="x", arity=1, fn=None)

    primitive = equation_ast_module.Primitive(name="x", arity=2, fn=lambda values: values[0] + values[1])
    with pytest.raises(ValueError, match="expects 2 operand"):
        primitive.apply([1.0])

    with pytest.raises(TypeError, match="value must be a real scalar"):
        Constant("bad")
    with pytest.raises(ValueError, match="value must be finite"):
        Constant(float("nan"))
    assert str(Constant(1.25)) == "1.25"

    with pytest.raises(TypeError, match="index must be an integer"):
        Variable(index=1.5)
    with pytest.raises(ValueError, match="index must be >="):
        Variable(index=-1)
    with pytest.raises(TypeError, match="name must be a string or None"):
        Variable(index=0, name=1)
    with pytest.raises(ValueError, match="name must be non-empty"):
        Variable(index=0, name="")
    with pytest.raises(TypeError, match="name must be a string"):
        Variable.from_name(1, ["x"])
    with pytest.raises(ValueError, match="name must be non-empty"):
        Variable.from_name("", ["x"])
    with pytest.raises(ValueError, match="unknown variable name"):
        Variable.from_name("y", ["x"])
    with pytest.raises(ValueError, match="out of range"):
        Variable(index=2).evaluate(np.array([1.0]))
    assert str(Variable(index=0, name="velocity")) == "velocity"
    assert str(Variable(index=1)) == "x[1]"

    with pytest.raises(TypeError, match="name must be a string"):
        Operator(name=1, operands=(Constant(1.0),))
    with pytest.raises(ValueError, match="name must be non-empty"):
        Operator(name="", operands=(Constant(1.0),))
    with pytest.raises(ValueError, match="unknown operator"):
        Operator(name="unknown", operands=(Constant(1.0),))
    with pytest.raises(TypeError, match="operands must be a tuple"):
        Operator(name="+", operands=[Constant(1.0), Constant(2.0)])
    with pytest.raises(ValueError, match="expects 2 operand"):
        Operator(name="+", operands=(Constant(1.0),))
    with pytest.raises(TypeError, match="operands must all be ASTNode"):
        Operator(name="+", operands=(Constant(1.0), 2.0))

    assert str(Operator("neg", (Variable(0),))) == "(-x[0])"
    assert str(Operator("+", (Variable(0), Constant(1.0)))) == "(x[0] + 1)"
    assert str(Operator("safe_pow", (Variable(0), Constant(2.0)))) == "(x[0] ** 2)"
    assert str(Operator("sin", (Variable(0),))) == "sin(x[0])"


def test_dataclass_bool_wrappers_and_equation_ast_validation():
    assert bool(equation_ast_module.DimensionValidation(True))
    assert bool(equation_ast_module.NumericalValidation(True))
    assert bool(equation_ast_module.StabilityReport(True, 0.0, 1.0))

    with pytest.raises(TypeError, match="root must be an ASTNode"):
        EquationAST(root="bad")
    with pytest.raises(ValueError, match="fitness must be finite"):
        EquationAST(root=Constant(1.0), fitness=float("nan"))
    with pytest.raises(TypeError, match="parent must be a string or None"):
        EquationAST(root=Constant(1.0), parent=1)
    with pytest.raises(TypeError, match="co_parent must be a string or None"):
        EquationAST(root=Constant(1.0), co_parent=1)
    with pytest.raises(TypeError, match="generation must be an integer"):
        EquationAST(root=Constant(1.0), generation=1.5)
    with pytest.raises(ValueError, match="generation must be >="):
        EquationAST(root=Constant(1.0), generation=-1)
    with pytest.raises(TypeError, match="id must be a string"):
        EquationAST(root=Constant(1.0), id=1)
    with pytest.raises(ValueError, match="id must be non-empty"):
        EquationAST(root=Constant(1.0), id="")

    equation = EquationAST(root=Variable(0))
    assert equation.evaluate(np.array([3.0])) == pytest.approx(3.0)


def test_subtree_operations_and_simplification_cover_error_and_edge_paths(monkeypatch):
    tree = Operator("+", (Variable(0), Constant(1.0)))
    with pytest.raises(IndexError, match="invalid subtree path"):
        subtree_at(tree, (2,))
    with pytest.raises(TypeError, match="replacement must be an ASTNode"):
        replace_subtree(tree, (), "bad")
    with pytest.raises(IndexError, match="invalid subtree path"):
        replace_subtree(Constant(1.0), (0,), Constant(2.0))
    with pytest.raises(IndexError, match="invalid subtree path"):
        replace_subtree(tree, (5,), Constant(2.0))

    broken_primitive = equation_ast_module.Primitive(name="broken", arity=2, fn=lambda values: 1.0 / 0.0)
    broken = Operator(
        name="broken",
        operands=(Constant(1.0), Constant(2.0)),
        library={"broken": broken_primitive},
    )
    assert simplify_ast(broken) == broken
    assert simplify_ast(Operator("+", (Constant(0.0), Variable(0)))) == Variable(0)
    assert simplify_ast(Operator("-", (Variable(0), Constant(0.0)))) == Variable(0)
    assert simplify_ast(Operator("/", (Constant(0.0), Variable(0)))) == Constant(0.0)
    assert simplify_ast(Operator("/", (Variable(0), Constant(1.0)))) == Variable(0)
    assert simplify_ast(Operator("pow", (Variable(0), Constant(1.0)))) == Variable(0)
    assert simplify_ast(Operator("pow", (Variable(0), Constant(0.0)))) == Constant(1.0)
    assert simplify_ast(Operator("pow", (Constant(1.0), Variable(0)))) == Constant(1.0)


def test_forge_constructor_and_dimension_helpers_validate_arguments():
    with pytest.raises(TypeError, match="population_size must be an integer"):
        EquationASTForge(population_size=1.5)
    with pytest.raises(ValueError, match="population_size must be positive"):
        EquationASTForge(population_size=0)
    with pytest.raises(ValueError, match="elite_fraction must be in"):
        EquationASTForge(elite_fraction=2.0)
    with pytest.raises(TypeError, match="max_depth must be an integer"):
        EquationASTForge(max_depth=1.5)
    with pytest.raises(ValueError, match="max_depth must be >="):
        EquationASTForge(max_depth=0)
    with pytest.raises(ValueError, match="constant_scale must be positive and finite"):
        EquationASTForge(constant_scale=0.0)

    forge = EquationASTForge()
    with pytest.raises(ValueError, match="max_depth must be >="):
        forge.random_tree(dim=1, max_depth=0)
    with pytest.raises(TypeError, match="dim must be an integer"):
        forge._validate_dim_and_names(1.5, None)
    with pytest.raises(ValueError, match="dim must be positive"):
        forge._validate_dim_and_names(0, None)
    with pytest.raises(ValueError, match="exactly 2 names"):
        forge._validate_dim_and_names(2, ["x"])
    with pytest.raises(ValueError, match="non-empty strings"):
        forge._validate_dim_and_names(1, [""])
    with pytest.raises(ValueError, match="must both have shape"):
        forge._validate_bounds(np.array([0.0]), np.array([1.0, 2.0]), 1)
    with pytest.raises(ValueError, match="lower must be <="):
        forge._validate_bounds(np.array([2.0]), np.array([1.0]), 1)


def test_forge_mutation_crossover_and_validation_error_paths(monkeypatch):
    forge = EquationASTForge(max_depth=1, primitives={"neg": equation_ast_module.Primitive("neg", 1, equation_ast_module._raw_neg)})

    with pytest.raises(TypeError, match="equation must be an EquationAST"):
        forge.mutate("bad", dim=1)
    with pytest.raises(TypeError, match="node must be an ASTNode"):
        forge.mutate_tree("bad", dim=1)

    original_random_tree = forge.random_tree

    class _MutateRng:
        def integers(self, low, high=None):
            return 0

        def random(self):
            return 0.9

        def normal(self, loc=0.0, scale=1.0, size=None):
            return np.zeros(size if size is not None else (), dtype=float) + loc

    monkeypatch.setattr(
        forge,
        "random_tree",
        lambda dim, max_depth=None, variable_names=None: Constant(0.0)
        if max_depth == forge.max_depth
        else Operator("neg", (Constant(1.0),), forge.primitives),
    )
    monkeypatch.setattr(forge, "rng", _MutateRng())
    mutated = forge.mutate_tree(Constant(1.0), dim=1)
    assert mutated == Constant(0.0)

    single_op_forge = EquationASTForge(
        max_depth=3,
        primitives={"neg": equation_ast_module.Primitive("neg", 1, equation_ast_module._raw_neg)},
    )

    class _SingleOpRng:
        def integers(self, low, high=None):
            return 0

        def random(self):
            return 0.1

        def normal(self, loc=0.0, scale=1.0, size=None):
            return np.zeros(size if size is not None else (), dtype=float) + loc

    monkeypatch.setattr(single_op_forge, "random_tree", lambda dim, max_depth=None, variable_names=None: Constant(2.0))
    monkeypatch.setattr(single_op_forge, "rng", _SingleOpRng())
    replaced = single_op_forge.mutate_tree(Operator("neg", (Constant(1.0),), single_op_forge.primitives), dim=1)
    assert replaced == Constant(2.0)

    with pytest.raises(TypeError, match="parent_a must be an EquationAST"):
        forge.crossover("bad", EquationAST(Constant(1.0)))
    with pytest.raises(TypeError, match="parent_b must be an EquationAST"):
        forge.crossover(EquationAST(Constant(1.0)), "bad")
    with pytest.raises(TypeError, match="recipient must be an ASTNode"):
        forge.crossover_tree("bad", Constant(1.0))
    with pytest.raises(TypeError, match="donor must be an ASTNode"):
        forge.crossover_tree(Constant(1.0), "bad")

    monkeypatch.setattr(equation_ast_module, "iter_subtree_paths", lambda node, path=(): ((),))
    monkeypatch.setattr(
        equation_ast_module,
        "subtree_at",
        lambda node, path: Operator("neg", (Constant(1.0),))
        if isinstance(node, Operator)
        else node,
    )
    assert forge.crossover_tree(Constant(1.0), Operator("neg", (Constant(1.0),))) == Constant(-1.0)

    monkeypatch.setattr(forge, "random_tree", original_random_tree)
    with pytest.raises(TypeError, match="node must be an ASTNode"):
        forge.simplify("bad")
    with pytest.raises(TypeError, match="node must be an ASTNode"):
        forge.validate_dimensions("bad", 1)
    with pytest.raises(TypeError, match="dim must be an integer"):
        forge.validate_dimensions(Constant(1.0), 1.5)
    with pytest.raises(ValueError, match="dim must be positive"):
        forge.validate_dimensions(Constant(1.0), 0)
    with pytest.raises(TypeError, match="node must be an ASTNode"):
        forge.validate_numerically("bad", np.zeros((1, 1)))
    with pytest.raises(ValueError, match="samples must be a 2D array"):
        forge.validate_numerically(Constant(1.0), np.zeros(1))


def test_stability_and_evolution_edge_cases(monkeypatch):
    forge = EquationASTForge(population_size=6, rng=np.random.default_rng(0))

    with pytest.raises(TypeError, match="node must be an ASTNode"):
        forge.analyze_stability("bad", np.zeros((1, 1)))
    with pytest.raises(ValueError, match="states must be a 2D array"):
        forge.analyze_stability(Constant(1.0), np.zeros(1))
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        forge.analyze_stability(Constant(1.0), np.zeros((1, 1)), epsilon=0.0)
    with pytest.raises(ValueError, match="threshold must be positive and finite"):
        forge.analyze_stability(Constant(1.0), np.zeros((1, 1)), threshold=0.0)

    nonfinite_baseline = Operator("log", (Constant(-1.0),))
    baseline_report = forge.analyze_stability(nonfinite_baseline, np.array([[0.0]]))
    assert not baseline_report.passed

    shifted_nonfinite = Operator("/", (Constant(1.0), Variable(0)))
    shifted_report = forge.analyze_stability(shifted_nonfinite, np.array([[1e-4]]), epsilon=-1e-4 + 2e-4)
    assert not shifted_report.passed

    with pytest.raises(TypeError, match="score_fn must be callable"):
        forge.evolve(dim=1, score_fn=None, lower=np.array([0.0]), upper=np.array([1.0]))
    with pytest.raises(TypeError, match="generations must be an integer"):
        forge.evolve(dim=1, score_fn=lambda node: 0.0, lower=np.array([0.0]), upper=np.array([1.0]), generations=1.5)
    with pytest.raises(ValueError, match="generations must be positive"):
        forge.evolve(dim=1, score_fn=lambda node: 0.0, lower=np.array([0.0]), upper=np.array([1.0]), generations=0)
    with pytest.raises(TypeError, match="sample_count must be an integer"):
        forge.evolve(dim=1, score_fn=lambda node: 0.0, lower=np.array([0.0]), upper=np.array([1.0]), sample_count=1.5)
    with pytest.raises(ValueError, match="sample_count must be positive"):
        forge.evolve(dim=1, score_fn=lambda node: 0.0, lower=np.array([0.0]), upper=np.array([1.0]), sample_count=0)
    with pytest.raises(ValueError, match="objective_mode must be"):
        forge.evolve(dim=1, score_fn=lambda node: 0.0, lower=np.array([0.0]), upper=np.array([1.0]), objective_mode="median")
    with pytest.raises(ValueError, match="complexity_penalty must be finite and >="):
        forge.evolve(dim=1, score_fn=lambda node: 0.0, lower=np.array([0.0]), upper=np.array([1.0]), complexity_penalty=-1.0)

    repopulating_forge = EquationASTForge(population_size=2, rng=np.random.default_rng(0))
    valid_equation = EquationAST(root=Constant(1.0), fitness=1.0, objective=1.0)
    rank_calls = {"count": 0}

    monkeypatch.setattr(
        repopulating_forge,
        "_initial_population",
        lambda dim, variable_names=None, seeds=None: [EquationAST(root=Constant(0.0))],
    )

    def rank_population(population, **kwargs):
        rank_calls["count"] += 1
        return [] if rank_calls["count"] == 1 else [valid_equation]

    monkeypatch.setattr(repopulating_forge, "_rank_population", rank_population)
    monkeypatch.setattr(repopulating_forge, "_next_generation", lambda ranked, **kwargs: list(ranked))
    repopulated = repopulating_forge.evolve(
        dim=1,
        score_fn=lambda node: 0.0,
        lower=np.array([0.0]),
        upper=np.array([1.0]),
        generations=2,
    )
    assert repopulated.best_equation == valid_equation

    failing_forge = EquationASTForge(population_size=2, rng=np.random.default_rng(0))
    monkeypatch.setattr(
        failing_forge,
        "_initial_population",
        lambda dim, variable_names=None, seeds=None: [EquationAST(root=Constant(0.0))],
    )
    monkeypatch.setattr(failing_forge, "_rank_population", lambda population, **kwargs: [])
    with pytest.raises(RuntimeError, match="no valid equation survived"):
        failing_forge.evolve(
            dim=1,
            score_fn=lambda node: 0.0,
            lower=np.array([0.0]),
            upper=np.array([1.0]),
            generations=1,
        )


def test_initial_population_and_ranking_cover_seed_and_filter_paths():
    forge = EquationASTForge(population_size=10, rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="seeds must contain only ASTNode"):
        forge._initial_population(dim=1, variable_names=None, seeds=[Constant(0.0), "bad"])

    population = forge._initial_population(dim=2, variable_names=["x0", "x1"], seeds=None)
    assert any(isinstance(candidate.root, Constant) and candidate.root.value == 0.0 for candidate in population)
    assert any(isinstance(candidate.root, Variable) and candidate.root.name == "x0" for candidate in population)
    assert len(population) == forge.population_size

    invalid = EquationAST(root=Variable(2))
    ranked = forge._rank_population(
        [invalid],
        dim=2,
        score_fn=lambda node: 0.0,
        samples=np.zeros((2, 2)),
        stability_states=np.zeros((1, 2)),
        stability_threshold=10.0,
        stability_epsilon=1e-4,
        objective_mode="min",
        complexity_penalty=0.0,
    )
    assert ranked == []
