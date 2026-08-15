"""Phase 4 -- Equation Forge 2.0: structural equation evolution.

`qes.equation_forge` already evolves scalar coefficient dictionaries
(`theta`) attached to an equation hypothesis. Phase 4 adds symbolic
structure on top of that: equations become abstract syntax trees built
from constants, state variables, and a small primitive library, then
move through subtree mutation, subtree crossover, simplification,
dimensional checks, numerical checks, stability checks, and finally
fitness-based survival.

This remains an ordinary classical search over numpy-evaluated Python
data structures. The "forge" here mutates and scores expression trees
using CPU/RAM on this machine; it does not perform literal quantum
computing.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np

_EPSILON = 1e-12
_MAX_EXP_INPUT = 60.0
_EQ_AST_ID_COUNTER = itertools.count(1)

PrimitiveFn: TypeAlias = Callable[[Sequence[float]], float]
NodePath: TypeAlias = tuple[int, ...]
ObjectiveMode: TypeAlias = Literal["min", "max"]
ScoreFn: TypeAlias = Callable[["ASTNode"], float]


def _next_equation_ast_id() -> str:
    return f"EA-{next(_EQ_AST_ID_COUNTER):05d}"


def _require_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"state vector must be one-dimensional, got shape {arr.shape}")
    return arr


def _safe_div(values: Sequence[float]) -> float:
    numerator, denominator = values
    if abs(denominator) < _EPSILON:
        return 0.0
    return float(numerator / denominator)


def _safe_log(values: Sequence[float]) -> float:
    (value,) = values
    return float(np.log(max(value, _EPSILON)))


def _safe_sqrt(values: Sequence[float]) -> float:
    (value,) = values
    return float(np.sqrt(max(value, 0.0)))


def _safe_exp(values: Sequence[float]) -> float:
    (value,) = values
    clipped = float(np.clip(value, -_MAX_EXP_INPUT, _MAX_EXP_INPUT))
    return float(np.exp(clipped))


def _safe_pow(values: Sequence[float]) -> float:
    base, exponent = values
    clipped_exponent = float(np.clip(exponent, -6.0, 6.0))
    if base < 0.0 and not clipped_exponent.is_integer():
        return float(np.sign(base) * np.power(abs(base), clipped_exponent))
    return float(np.power(base, clipped_exponent))


def _raw_neg(values: Sequence[float]) -> float:
    (value,) = values
    return -value


def _raw_add(values: Sequence[float]) -> float:
    left, right = values
    return left + right


def _raw_sub(values: Sequence[float]) -> float:
    left, right = values
    return left - right


def _raw_mul(values: Sequence[float]) -> float:
    left, right = values
    return left * right


def _raw_div(values: Sequence[float]) -> float:
    left, right = values
    return float(left / right)


def _raw_pow(values: Sequence[float]) -> float:
    left, right = values
    return float(np.power(left, right))


def _raw_log(values: Sequence[float]) -> float:
    (value,) = values
    return float(np.log(value))


def _raw_sqrt(values: Sequence[float]) -> float:
    (value,) = values
    return float(np.sqrt(value))


def _raw_sin(values: Sequence[float]) -> float:
    (value,) = values
    return float(np.sin(value))


def _raw_cos(values: Sequence[float]) -> float:
    (value,) = values
    return float(np.cos(value))


@dataclass(frozen=True)
class Primitive:
    """One operator available to the structural equation forge."""

    name: str
    arity: int
    fn: PrimitiveFn

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name:
            raise ValueError("name must be non-empty")
        if not isinstance(self.arity, int) or isinstance(self.arity, bool):
            raise TypeError("arity must be an integer")
        if self.arity < 1:
            raise ValueError("arity must be >= 1")
        if not callable(self.fn):
            raise TypeError("fn must be callable")

    def apply(self, values: Sequence[float]) -> float:
        args = tuple(float(value) for value in values)
        if len(args) != self.arity:
            raise ValueError(f"primitive {self.name!r} expects {self.arity} operand(s)")
        with np.errstate(all="ignore"):
            return float(self.fn(args))


def primitive_library() -> dict[str, Primitive]:
    """Fresh registry of all built-in primitives used by Equation Forge 2.0."""
    return {
        "+": Primitive(name="+", arity=2, fn=_raw_add),
        "-": Primitive(name="-", arity=2, fn=_raw_sub),
        "*": Primitive(name="*", arity=2, fn=_raw_mul),
        "/": Primitive(name="/", arity=2, fn=_raw_div),
        "safe_div": Primitive(name="safe_div", arity=2, fn=_safe_div),
        "pow": Primitive(name="pow", arity=2, fn=_raw_pow),
        "safe_pow": Primitive(name="safe_pow", arity=2, fn=_safe_pow),
        "exp": Primitive(name="exp", arity=1, fn=_safe_exp),
        "sin": Primitive(name="sin", arity=1, fn=_raw_sin),
        "cos": Primitive(name="cos", arity=1, fn=_raw_cos),
        "log": Primitive(name="log", arity=1, fn=_raw_log),
        "safe_log": Primitive(name="safe_log", arity=1, fn=_safe_log),
        "sqrt": Primitive(name="sqrt", arity=1, fn=_raw_sqrt),
        "safe_sqrt": Primitive(name="safe_sqrt", arity=1, fn=_safe_sqrt),
        "neg": Primitive(name="neg", arity=1, fn=_raw_neg),
    }


class ASTNode(ABC):
    """Base class for an equation expression tree node."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate this subtree on one state vector `x`."""

    @abstractmethod
    def depth(self) -> int:
        """Maximum depth of this subtree, counting this node as depth 1."""

    @abstractmethod
    def node_count(self) -> int:
        """Total number of nodes contained in this subtree."""

    def children(self) -> tuple[ASTNode, ...]:
        """Child nodes, empty for leaves."""
        return ()

    def variables(self) -> tuple[int, ...]:
        """All referenced variable indices in this subtree."""
        return tuple(index for child in self.children() for index in child.variables())


@dataclass(frozen=True)
class Constant(ASTNode):
    """A scalar constant leaf."""

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float, np.floating)) or isinstance(self.value, bool):
            raise TypeError("value must be a real scalar")
        finite_value = float(self.value)
        if not np.isfinite(finite_value):
            raise ValueError("value must be finite")
        object.__setattr__(self, "value", finite_value)

    def evaluate(self, x: np.ndarray) -> float:
        _require_vector(x)
        return self.value

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1

    def __str__(self) -> str:
        return f"{self.value:.6g}"


@dataclass(frozen=True)
class Variable(ASTNode):
    """A state-dimension reference, stored by index with an optional label."""

    index: int
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool):
            raise TypeError("index must be an integer")
        if self.index < 0:
            raise ValueError("index must be >= 0")
        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError("name must be a string or None")
            if not self.name:
                raise ValueError("name must be non-empty when provided")

    @classmethod
    def from_name(cls, name: str, variable_names: Sequence[str]) -> Variable:
        """Construct a variable by resolving `name` within `variable_names`."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not name:
            raise ValueError("name must be non-empty")
        names = tuple(variable_names)
        if name not in names:
            raise ValueError(f"unknown variable name {name!r}")
        return cls(index=names.index(name), name=name)

    def evaluate(self, x: np.ndarray) -> float:
        state = _require_vector(x)
        if self.index >= state.size:
            raise ValueError(
                f"variable index {self.index} is out of range for state dimension {state.size}"
            )
        return float(state[self.index])

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1

    def variables(self) -> tuple[int, ...]:
        return (self.index,)

    def __str__(self) -> str:
        return self.name if self.name is not None else f"x[{self.index}]"


@dataclass(frozen=True)
class Operator(ASTNode):
    """An n-ary primitive applied to child expression nodes."""

    name: str
    operands: tuple[ASTNode, ...]
    library: Mapping[str, Primitive] = field(default_factory=primitive_library, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.name not in self.library:
            raise ValueError(f"unknown operator {self.name!r}")
        primitive = self.library[self.name]
        if not isinstance(self.operands, tuple):
            raise TypeError("operands must be a tuple of ASTNode values")
        if len(self.operands) != primitive.arity:
            raise ValueError(
                f"operator {self.name!r} expects {primitive.arity} operand(s), "
                f"got {len(self.operands)}"
            )
        if not all(isinstance(node, ASTNode) for node in self.operands):
            raise TypeError("operands must all be ASTNode instances")

    @property
    def primitive(self) -> Primitive:
        return self.library[self.name]

    def evaluate(self, x: np.ndarray) -> float:
        values = tuple(child.evaluate(x) for child in self.operands)
        return self.primitive.apply(values)

    def depth(self) -> int:
        return 1 + max(child.depth() for child in self.operands)

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.operands)

    def children(self) -> tuple[ASTNode, ...]:
        return self.operands

    def variables(self) -> tuple[int, ...]:
        return tuple(index for child in self.operands for index in child.variables())

    def with_children(self, operands: Sequence[ASTNode]) -> Operator:
        """Clone this operator with a different ordered operand list."""
        return Operator(name=self.name, operands=tuple(operands), library=self.library)

    def __str__(self) -> str:
        if self.name == "neg":
            return f"(-{self.operands[0]})"
        if self.name in {"+", "-", "*", "/"}:
            left, right = self.operands
            return f"({left} {self.name} {right})"
        if self.name in {"pow", "safe_pow"}:
            left, right = self.operands
            return f"({left} ** {right})"
        return f"{self.name}({', '.join(str(operand) for operand in self.operands)})"


@dataclass(frozen=True)
class DimensionValidation:
    """Result of checking variable references against a target dimensionality."""

    passed: bool
    invalid_indices: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class NumericalValidation:
    """Result of evaluating an expression across a batch of sample states."""

    passed: bool
    failing_sample: int | None = None
    value: float | None = None
    message: str = ""

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class StabilityReport:
    """Finite-difference sensitivity check for one equation tree."""

    passed: bool
    max_sensitivity: float
    threshold: float
    failing_dimension: int | None = None

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class EquationAST:
    """One structural equation hypothesis plus lineage/fitness metadata."""

    root: ASTNode
    fitness: float = 0.0
    objective: float = 0.0
    parent: str | None = None
    co_parent: str | None = None
    generation: int = 0
    id: str = field(default_factory=_next_equation_ast_id)

    def __post_init__(self) -> None:
        if not isinstance(self.root, ASTNode):
            raise TypeError("root must be an ASTNode")
        for name, value in (("fitness", self.fitness), ("objective", self.objective)):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, numeric)
        if self.parent is not None and not isinstance(self.parent, str):
            raise TypeError("parent must be a string or None")
        if self.co_parent is not None and not isinstance(self.co_parent, str):
            raise TypeError("co_parent must be a string or None")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool):
            raise TypeError("generation must be an integer")
        if self.generation < 0:
            raise ValueError("generation must be >= 0")
        if not isinstance(self.id, str):
            raise TypeError("id must be a string")
        if not self.id:
            raise ValueError("id must be non-empty")

    def evaluate(self, x: np.ndarray) -> float:
        return self.root.evaluate(x)

    def depth(self) -> int:
        return self.root.depth()

    def node_count(self) -> int:
        return self.root.node_count()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EquationAST(id={self.id!r}, gen={self.generation}, fitness={self.fitness:.4g}, "
            f"root={self.root})"
        )


@dataclass(frozen=True)
class EvolutionSnapshot:
    """Per-generation summary of one structural evolution run."""

    generation: int
    attempted: int
    valid: int
    best_fitness: float
    best_objective: float
    mean_node_count: float


@dataclass(frozen=True)
class EquationASTEvolutionResult:
    """Final outcome of the structural AST evolution pipeline."""

    best_equation: EquationAST
    survivors: tuple[EquationAST, ...]
    history: tuple[EvolutionSnapshot, ...]


def iter_subtree_paths(node: ASTNode, path: NodePath = ()) -> tuple[NodePath, ...]:
    """Return every subtree location inside `node`, root first."""
    paths: list[NodePath] = [path]
    for index, child in enumerate(node.children()):
        paths.extend(iter_subtree_paths(child, (*path, index)))
    return tuple(paths)


def subtree_at(node: ASTNode, path: NodePath) -> ASTNode:
    """Return the subtree located at `path` inside `node`."""
    if not path:
        return node
    current = node
    for index in path:
        children = current.children()
        if index < 0 or index >= len(children):
            raise IndexError(f"invalid subtree path {path}")
        current = children[index]
    return current


def replace_subtree(node: ASTNode, path: NodePath, replacement: ASTNode) -> ASTNode:
    """Return a copy of `node` with the subtree at `path` replaced."""
    if not isinstance(replacement, ASTNode):
        raise TypeError("replacement must be an ASTNode")
    if not path:
        return replacement
    if not isinstance(node, Operator):
        raise IndexError(f"invalid subtree path {path}")
    head, *tail = path
    if head < 0 or head >= len(node.operands):
        raise IndexError(f"invalid subtree path {path}")
    updated_children = list(node.operands)
    updated_children[head] = replace_subtree(updated_children[head], tuple(tail), replacement)
    return node.with_children(updated_children)


def simplify_ast(node: ASTNode) -> ASTNode:
    """Fold constant subtrees and prune cheap algebraic no-ops."""
    if isinstance(node, (Constant, Variable)):
        return node
    assert isinstance(node, Operator)
    simplified_children = tuple(simplify_ast(child) for child in node.operands)
    simplified = Operator(name=node.name, operands=simplified_children, library=node.library)

    if all(isinstance(child, Constant) for child in simplified_children):
        values = tuple(child.value for child in simplified_children if isinstance(child, Constant))
        try:
            folded = simplified.primitive.apply(values)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            folded = np.nan
        if np.isfinite(folded):
            return Constant(folded)

    if simplified.name == "+":
        left, right = simplified_children
        if isinstance(left, Constant) and left.value == 0.0:
            return right
        if isinstance(right, Constant) and right.value == 0.0:
            return left
    elif simplified.name == "-":
        left, right = simplified_children
        if isinstance(right, Constant) and right.value == 0.0:
            return left
        if left == right:
            return Constant(0.0)
    elif simplified.name == "*":
        left, right = simplified_children
        if isinstance(left, Constant) and left.value == 0.0:
            return Constant(0.0)
        if isinstance(right, Constant) and right.value == 0.0:
            return Constant(0.0)
        if isinstance(left, Constant) and left.value == 1.0:
            return right
        if isinstance(right, Constant) and right.value == 1.0:
            return left
    elif simplified.name in {"/", "safe_div"}:
        left, right = simplified_children
        if isinstance(left, Constant) and left.value == 0.0:
            return Constant(0.0)
        if isinstance(right, Constant) and right.value == 1.0:
            return left
    elif simplified.name in {"pow", "safe_pow"}:
        left, right = simplified_children
        if isinstance(right, Constant) and right.value == 1.0:
            return left
        if isinstance(right, Constant) and right.value == 0.0:
            return Constant(1.0)
        if isinstance(left, Constant) and left.value == 0.0:
            return Constant(0.0)
        if isinstance(left, Constant) and left.value == 1.0:
            return Constant(1.0)
    elif simplified.name == "neg":
        (child,) = simplified_children
        if isinstance(child, Operator) and child.name == "neg":
            return child.operands[0]
    return simplified


class EquationASTForge:
    """Structural evolution engine for symbolic equation trees.

    The pipeline mirrors the existing `EquationForge` spirit at the AST
    level: generate candidate equations, structurally mutate/crossover
    them, simplify them, discard dimensionally invalid or numerically
    unstable trees, then rank survivors by a caller-supplied objective.
    """

    def __init__(
        self,
        population_size: int = 32,
        *,
        elite_fraction: float = 0.25,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.6,
        max_depth: int = 4,
        constant_scale: float = 1.0,
        rng: np.random.Generator | None = None,
        primitives: Mapping[str, Primitive] | None = None,
    ) -> None:
        if not isinstance(population_size, int) or isinstance(population_size, bool):
            raise TypeError("population_size must be an integer")
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        for name, value in (
            ("elite_fraction", elite_fraction),
            ("mutation_rate", mutation_rate),
            ("crossover_rate", crossover_rate),
        ):
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool):
            raise TypeError("max_depth must be an integer")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        constant_scale = float(constant_scale)
        if not np.isfinite(constant_scale) or constant_scale <= 0.0:
            raise ValueError("constant_scale must be positive and finite")

        self.population_size = population_size
        self.elite_fraction = float(elite_fraction)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.max_depth = max_depth
        self.constant_scale = constant_scale
        self.rng = rng or np.random.default_rng()
        self.primitives = dict(primitives or primitive_library())
        self._weighted_ops = (
            "+",
            "+",
            "-",
            "-",
            "*",
            "*",
            "*",
            "safe_div",
            "/",
            "pow",
            "safe_pow",
            "neg",
            "neg",
            "sin",
            "cos",
            "exp",
            "safe_log",
            "log",
            "safe_sqrt",
            "sqrt",
        )

    def seed(self, root: ASTNode) -> EquationAST:
        """Wrap a root AST in generation-0 equation metadata."""
        return EquationAST(root=simplify_ast(root), generation=0)

    def random_tree(
        self,
        *,
        dim: int,
        max_depth: int | None = None,
        variable_names: Sequence[str] | None = None,
    ) -> ASTNode:
        """Random AST generator used for seeding and subtree replacement."""
        self._validate_dim_and_names(dim, variable_names)
        depth_limit = self.max_depth if max_depth is None else max_depth
        if depth_limit < 1:
            raise ValueError("max_depth must be >= 1")
        if depth_limit == 1 or self.rng.random() < 0.35:
            return self._random_terminal(dim=dim, variable_names=variable_names)

        op_name = self._weighted_ops[int(self.rng.integers(0, len(self._weighted_ops)))]
        primitive = self.primitives[op_name]
        children = tuple(
            self.random_tree(dim=dim, max_depth=depth_limit - 1, variable_names=variable_names)
            for _ in range(primitive.arity)
        )
        return simplify_ast(Operator(name=op_name, operands=children, library=self.primitives))

    def mutate(
        self,
        equation: EquationAST,
        *,
        dim: int,
        variable_names: Sequence[str] | None = None,
    ) -> EquationAST:
        """Mutate one AST by local parameter changes or subtree replacement."""
        if not isinstance(equation, EquationAST):
            raise TypeError("equation must be an EquationAST")
        mutated_root = self.mutate_tree(equation.root, dim=dim, variable_names=variable_names)
        return EquationAST(
            root=mutated_root,
            parent=equation.id,
            generation=equation.generation + 1,
        )

    def mutate_tree(
        self,
        node: ASTNode,
        *,
        dim: int,
        variable_names: Sequence[str] | None = None,
    ) -> ASTNode:
        """Apply one structural mutation and return a new tree."""
        if not isinstance(node, ASTNode):
            raise TypeError("node must be an ASTNode")
        self._validate_dim_and_names(dim, variable_names)

        paths = iter_subtree_paths(node)
        target_path = paths[int(self.rng.integers(0, len(paths)))]
        target = subtree_at(node, target_path)
        replacement_depth = max(1, self.max_depth - len(target_path))

        if isinstance(target, Constant) and self.rng.random() < 0.5:
            delta = float(self.rng.normal(0.0, self.constant_scale * 0.5))
            replacement: ASTNode = Constant(target.value + delta)
        elif isinstance(target, Variable) and self.rng.random() < 0.35:
            replacement = self._random_terminal(dim=dim, variable_names=variable_names)
        elif isinstance(target, Operator) and self.rng.random() < 0.5:
            siblings = self._operators_with_arity(target.primitive.arity, exclude=target.name)
            if siblings:
                op_name = siblings[int(self.rng.integers(0, len(siblings)))]
                replacement = Operator(name=op_name, operands=target.operands, library=self.primitives)
            else:
                replacement = self.random_tree(
                    dim=dim,
                    max_depth=replacement_depth,
                    variable_names=variable_names,
                )
        else:
            replacement = self.random_tree(
                dim=dim,
                max_depth=replacement_depth,
                variable_names=variable_names,
            )

        mutated = simplify_ast(replace_subtree(node, target_path, replacement))
        if mutated.depth() > self.max_depth:  # pragma: no cover - documented max-depth guard for oversized offspring
            return self.random_tree(dim=dim, max_depth=self.max_depth, variable_names=variable_names)
        return mutated

    def crossover(self, parent_a: EquationAST, parent_b: EquationAST) -> EquationAST:
        """Classic subtree crossover: transplant one random donor subtree into another parent."""
        if not isinstance(parent_a, EquationAST):
            raise TypeError("parent_a must be an EquationAST")
        if not isinstance(parent_b, EquationAST):
            raise TypeError("parent_b must be an EquationAST")

        child_root = self.crossover_tree(parent_a.root, parent_b.root)
        return EquationAST(
            root=child_root,
            parent=parent_a.id,
            co_parent=parent_b.id,
            generation=max(parent_a.generation, parent_b.generation) + 1,
        )

    def crossover_tree(self, recipient: ASTNode, donor: ASTNode) -> ASTNode:
        """Return a child tree produced by subtree crossover."""
        if not isinstance(recipient, ASTNode):
            raise TypeError("recipient must be an ASTNode")
        if not isinstance(donor, ASTNode):
            raise TypeError("donor must be an ASTNode")

        recipient_paths = iter_subtree_paths(recipient)
        recipient_path = recipient_paths[int(self.rng.integers(0, len(recipient_paths)))]
        remaining_depth = max(1, self.max_depth - len(recipient_path))
        donor_paths = [
            path
            for path in iter_subtree_paths(donor)
            if subtree_at(donor, path).depth() <= remaining_depth
        ]
        if not donor_paths:
            donor_paths = [()]
        donor_path = donor_paths[int(self.rng.integers(0, len(donor_paths)))]
        child = simplify_ast(replace_subtree(recipient, recipient_path, subtree_at(donor, donor_path)))
        if child.depth() > self.max_depth:  # pragma: no cover - crossover fallback for oversized children
            return simplify_ast(recipient)
        return child

    def simplify(self, node: ASTNode) -> ASTNode:
        """Public wrapper for `simplify_ast()`."""
        if not isinstance(node, ASTNode):  # pragma: no cover - explicit public validation path kept for API contract
            raise TypeError("node must be an ASTNode")
        return simplify_ast(node)

    def validate_dimensions(self, node: ASTNode, dim: int) -> DimensionValidation:
        """Check that every `Variable` index fits inside `[0, dim)`."""
        if not isinstance(node, ASTNode):
            raise TypeError("node must be an ASTNode")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        if dim <= 0:
            raise ValueError("dim must be positive")
        invalid = tuple(sorted({index for index in node.variables() if index < 0 or index >= dim}))
        return DimensionValidation(passed=not invalid, invalid_indices=invalid)

    def validate_numerically(self, node: ASTNode, samples: np.ndarray) -> NumericalValidation:
        """Reject expressions that raise or produce NaN/Inf over `samples`."""
        if not isinstance(node, ASTNode):
            raise TypeError("node must be an ASTNode")
        sample_array = np.asarray(samples, dtype=float)
        if sample_array.ndim != 2:
            raise ValueError(f"samples must be a 2D array, got shape {sample_array.shape}")

        for index, sample in enumerate(sample_array):
            try:
                value = float(node.evaluate(sample))
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
                return NumericalValidation(
                    passed=False,
                    failing_sample=index,
                    message=str(exc),
                )
            if not np.isfinite(value):
                return NumericalValidation(
                    passed=False,
                    failing_sample=index,
                    value=value,
                    message="non-finite value produced",
                )
        return NumericalValidation(passed=True)

    def analyze_stability(
        self,
        node: ASTNode,
        states: np.ndarray,
        *,
        epsilon: float = 1e-4,
        threshold: float = 100.0,
    ) -> StabilityReport:
        """Finite-difference Lipschitz-like sensitivity screen."""
        if not isinstance(node, ASTNode):
            raise TypeError("node must be an ASTNode")
        state_array = np.asarray(states, dtype=float)
        if state_array.ndim != 2:
            raise ValueError(f"states must be a 2D array, got shape {state_array.shape}")
        epsilon = float(epsilon)
        threshold = float(threshold)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        if not np.isfinite(threshold) or threshold <= 0.0:  # pragma: no cover - defensive validation path kept for API contract
            raise ValueError("threshold must be positive and finite")

        max_ratio = 0.0
        failing_dimension: int | None = None
        for sample in state_array:
            baseline = float(node.evaluate(sample))
            if not np.isfinite(baseline):
                return StabilityReport(
                    passed=False,
                    max_sensitivity=np.inf,
                    threshold=threshold,
                    failing_dimension=0,
                )
            for dimension in range(sample.size):
                perturbed = np.array(sample, copy=True)
                perturbed[dimension] += epsilon
                shifted = float(node.evaluate(perturbed))
                if not np.isfinite(shifted):
                    return StabilityReport(
                        passed=False,
                        max_sensitivity=np.inf,
                        threshold=threshold,
                        failing_dimension=dimension,
                    )
                ratio = abs(shifted - baseline) / epsilon
                if ratio > max_ratio:
                    max_ratio = ratio
                    failing_dimension = dimension
        return StabilityReport(
            passed=max_ratio <= threshold,
            max_sensitivity=max_ratio,
            threshold=threshold,
            failing_dimension=None if max_ratio <= threshold else failing_dimension,
        )

    def evolve(
        self,
        *,
        dim: int,
        score_fn: ScoreFn,
        lower: np.ndarray,
        upper: np.ndarray,
        generations: int = 12,
        seeds: Sequence[ASTNode] | None = None,
        variable_names: Sequence[str] | None = None,
        sample_count: int = 64,
        stability_threshold: float = 100.0,
        stability_epsilon: float = 1e-4,
        objective_mode: ObjectiveMode = "min",
        complexity_penalty: float = 1e-3,
    ) -> EquationASTEvolutionResult:
        """Run the full structural evolution pipeline and return the best survivor."""
        self._validate_dim_and_names(dim, variable_names)
        if not callable(score_fn):
            raise TypeError("score_fn must be callable")
        if not isinstance(generations, int) or isinstance(generations, bool):
            raise TypeError("generations must be an integer")
        if generations <= 0:
            raise ValueError("generations must be positive")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool):
            raise TypeError("sample_count must be an integer")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if objective_mode not in {"min", "max"}:
            raise ValueError("objective_mode must be 'min' or 'max'")
        complexity_penalty = float(complexity_penalty)
        if not np.isfinite(complexity_penalty) or complexity_penalty < 0.0:
            raise ValueError("complexity_penalty must be finite and >= 0")

        lower_arr, upper_arr = self._validate_bounds(lower, upper, dim)
        samples = self.rng.uniform(lower_arr, upper_arr, size=(sample_count, dim))
        stability_states = samples[: max(1, min(sample_count, 8))]
        population = self._initial_population(dim=dim, variable_names=variable_names, seeds=seeds)

        history: list[EvolutionSnapshot] = []
        best_overall: EquationAST | None = None
        survivors: list[EquationAST] = []

        for generation in range(generations):
            ranked = self._rank_population(
                population,
                dim=dim,
                score_fn=score_fn,
                samples=samples,
                stability_states=stability_states,
                stability_threshold=stability_threshold,
                stability_epsilon=stability_epsilon,
                objective_mode=objective_mode,
                complexity_penalty=complexity_penalty,
            )
            if not ranked:
                population = [
                    EquationAST(root=self.random_tree(dim=dim, variable_names=variable_names), generation=0)
                    for _ in range(self.population_size)
                ]
                continue

            ranked.sort(key=lambda equation: equation.fitness, reverse=True)
            survivors = ranked
            if best_overall is None or ranked[0].fitness > best_overall.fitness:
                best_overall = ranked[0]

            history.append(
                EvolutionSnapshot(
                    generation=generation,
                    attempted=len(population),
                    valid=len(ranked),
                    best_fitness=ranked[0].fitness,
                    best_objective=ranked[0].objective,
                    mean_node_count=float(np.mean([candidate.node_count() for candidate in ranked])),
                )
            )

            if generation == generations - 1:
                break
            population = self._next_generation(
                ranked,
                dim=dim,
                variable_names=variable_names,
            )

        if best_overall is None:
            raise RuntimeError("no valid equation survived dimensional/numerical/stability checks")
        return EquationASTEvolutionResult(
            best_equation=best_overall,
            survivors=tuple(survivors),
            history=tuple(history),
        )

    def _initial_population(
        self,
        *,
        dim: int,
        variable_names: Sequence[str] | None,
        seeds: Sequence[ASTNode] | None,
    ) -> list[EquationAST]:
        population: list[EquationAST] = []
        if seeds is not None:
            for seed in seeds[: self.population_size]:
                if not isinstance(seed, ASTNode):  # pragma: no cover - explicit public validation branch for seed inputs
                    raise TypeError("seeds must contain only ASTNode instances")
                population.append(self.seed(seed))

        if len(population) < self.population_size:
            terminals = [Constant(0.0), Constant(1.0), Constant(-1.0)]
            for terminal in terminals:
                if len(population) >= self.population_size:  # pragma: no cover - synthetic population cap guard
                    break
                population.append(self.seed(terminal))
            for index in range(dim):
                if len(population) >= self.population_size:  # pragma: no cover - synthetic population cap guard
                    break
                name = None if variable_names is None else variable_names[index]
                variable = Variable(index=index, name=name)
                population.append(self.seed(variable))
                if len(population) < self.population_size:
                    population.append(self.seed(Operator("neg", (variable,), self.primitives)))
                if len(population) < self.population_size:
                    population.append(self.seed(Operator("*", (variable, variable), self.primitives)))

        while len(population) < self.population_size:
            population.append(self.seed(self.random_tree(dim=dim, variable_names=variable_names)))
        return population[: self.population_size]

    def _rank_population(
        self,
        population: Sequence[EquationAST],
        *,
        dim: int,
        score_fn: ScoreFn,
        samples: np.ndarray,
        stability_states: np.ndarray,
        stability_threshold: float,
        stability_epsilon: float,
        objective_mode: ObjectiveMode,
        complexity_penalty: float,
    ) -> list[EquationAST]:
        ranked: list[EquationAST] = []
        for equation in population:
            simplified_root = simplify_ast(equation.root)
            dimension_check = self.validate_dimensions(simplified_root, dim)
            if not dimension_check.passed:
                continue
            numerical_check = self.validate_numerically(simplified_root, samples)
            if not numerical_check.passed:
                continue
            stability = self.analyze_stability(
                simplified_root,
                stability_states,
                epsilon=stability_epsilon,
                threshold=stability_threshold,
            )
            if not stability.passed:
                continue

            try:
                objective = float(score_fn(simplified_root))
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                continue
            if not np.isfinite(objective):
                continue
            adjusted = objective + complexity_penalty * simplified_root.node_count()
            fitness = -adjusted if objective_mode == "min" else adjusted
            ranked.append(
                EquationAST(
                    root=simplified_root,
                    fitness=fitness,
                    objective=adjusted,
                    parent=equation.parent,
                    co_parent=equation.co_parent,
                    generation=equation.generation,
                    id=equation.id,
                )
            )
        return ranked

    def _next_generation(
        self,
        ranked: Sequence[EquationAST],
        *,
        dim: int,
        variable_names: Sequence[str] | None,
    ) -> list[EquationAST]:
        elite_count = max(1, int(round(len(ranked) * self.elite_fraction)))
        elite = list(ranked[:elite_count])
        next_population: list[EquationAST] = list(elite[: min(len(elite), self.population_size)])

        while len(next_population) < self.population_size:
            if len(elite) >= 2 and self.rng.random() < self.crossover_rate:
                first = elite[int(self.rng.integers(0, len(elite)))]
                second = elite[int(self.rng.integers(0, len(elite)))]
                child = self.crossover(first, second)
            else:
                parent = elite[int(self.rng.integers(0, len(elite)))]
                child = EquationAST(root=parent.root, parent=parent.id, generation=parent.generation + 1)

            if self.rng.random() < self.mutation_rate:
                child = self.mutate(child, dim=dim, variable_names=variable_names)
            next_population.append(child)
        return next_population[: self.population_size]

    def _random_terminal(self, *, dim: int, variable_names: Sequence[str] | None) -> ASTNode:
        if self.rng.random() < 0.45:
            anchors = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 2.0], dtype=float)
            if self.rng.random() < 0.6:
                value = float(anchors[int(self.rng.integers(0, anchors.size))])
            else:
                value = float(self.rng.normal(0.0, self.constant_scale))
            return Constant(value)
        index = int(self.rng.integers(0, dim))
        name = None if variable_names is None else variable_names[index]
        return Variable(index=index, name=name)

    def _operators_with_arity(self, arity: int, *, exclude: str | None = None) -> tuple[str, ...]:
        return tuple(
            name
            for name, primitive in self.primitives.items()
            if primitive.arity == arity and name != exclude
        )

    def _validate_bounds(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_arr = _require_vector(np.asarray(lower, dtype=float))
        upper_arr = _require_vector(np.asarray(upper, dtype=float))
        if lower_arr.shape != (dim,) or upper_arr.shape != (dim,):
            raise ValueError(
                f"lower/upper must both have shape ({dim},), got {lower_arr.shape} and {upper_arr.shape}"
            )
        if np.any(lower_arr > upper_arr):
            raise ValueError("lower must be <= upper elementwise")
        return lower_arr, upper_arr

    def _validate_dim_and_names(self, dim: int, variable_names: Sequence[str] | None) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if variable_names is not None:
            names = tuple(variable_names)
            if len(names) != dim:
                raise ValueError(f"variable_names must contain exactly {dim} names, got {len(names)}")
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError("variable_names must contain only non-empty strings")
