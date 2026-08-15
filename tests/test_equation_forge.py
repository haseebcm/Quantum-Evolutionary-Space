import numpy as np

from qes.equation_forge import EquationForge, seed_equation


def test_seed_equation_matches_formula():
    v = np.array([1.0, 2.0])
    interactions = np.eye(2)
    val = seed_equation(v, interactions, delta=1.0, lam=2.0)
    # Phi_I = v^T I v = 1*1 + 2*2 = 5 ; eps_0 = 2*5 - 1 = 9
    assert val == 9.0


def test_forge_seed_creates_generation_zero():
    forge = EquationForge(rng=np.random.default_rng(0))
    eq = forge.seed(theta={"a": 1.0})
    assert eq.generation == 0
    assert eq.parent is None


def test_mutate_increments_generation_and_tracks_parent():
    forge = EquationForge(rng=np.random.default_rng(0))
    seed = forge.seed(theta={"a": 1.0})
    child = forge.mutate(seed)
    assert child.generation == 1
    assert child.parent == seed.id


def test_mutate_preserves_non_numeric_theta():
    forge = EquationForge(rng=np.random.default_rng(0))
    seed = forge.seed(theta={"label": "quadratic", "a": 1.0})
    child = forge.mutate(seed)
    assert child.theta["label"] == "quadratic"


def test_spawn_population_creates_requested_count():
    forge = EquationForge(rng=np.random.default_rng(0))
    seed = forge.seed(theta={"a": 1.0})
    population = forge.spawn_population(seed, size=10)
    assert len(population) == 10
    assert all(eq.parent == seed.id for eq in population)


def test_suppress_filters_by_fitness():
    forge = EquationForge()
    seed = forge.seed()
    pop = forge.spawn_population(seed, 5)
    for i, eq in enumerate(pop):
        eq.fitness = i
    survivors = forge.suppress(pop, min_fitness=2)
    assert all(eq.fitness >= 2 for eq in survivors)
    assert len(survivors) == 3


def test_lineage_walks_back_to_seed():
    forge = EquationForge(rng=np.random.default_rng(0))
    seed = forge.seed(theta={"a": 1.0})
    child = forge.mutate(seed)
    grandchild = forge.mutate(child)
    registry = {seed.id: seed, child.id: child, grandchild.id: grandchild}
    chain = forge.lineage(grandchild, registry)
    assert [eq.id for eq in chain] == [seed.id, child.id, grandchild.id]


def test_crossover_blends_numeric_parameters_between_parents():
    forge = EquationForge(rng=np.random.default_rng(0))
    parent_a = forge.seed(theta={"a": 0.0})
    parent_b = forge.seed(theta={"a": 10.0})
    child = forge.crossover(parent_a, parent_b)
    assert 0.0 <= child.theta["a"] <= 10.0
    assert child.parent == parent_a.id
    assert child.theta["_co_parent"] == parent_b.id
    assert child.generation == 1


def test_crossover_preserves_non_numeric_keys_from_parent_a():
    forge = EquationForge(rng=np.random.default_rng(0))
    parent_a = forge.seed(theta={"label": "alpha", "a": 1.0})
    parent_b = forge.seed(theta={"label": "beta", "a": 3.0})
    child = forge.crossover(parent_a, parent_b)
    assert child.theta["label"] == "alpha"


def test_spawn_next_generation_preserves_elite_and_fills_remaining_slots():
    forge = EquationForge(rng=np.random.default_rng(0))
    seed = forge.seed(theta={"a": 1.0})
    population = forge.spawn_population(seed, size=10)
    for i, eq in enumerate(population):
        eq.fitness = float(i)

    next_gen = forge.spawn_next_generation(population, size=10, elite_fraction=0.2)
    assert len(next_gen) == 10
    best = max(population, key=lambda e: e.fitness)
    assert best in next_gen


def test_spawn_next_generation_handles_empty_population():
    forge = EquationForge()
    assert forge.spawn_next_generation([], size=5) == []
