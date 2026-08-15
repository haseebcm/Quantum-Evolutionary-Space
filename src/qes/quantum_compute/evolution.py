"""Evolution and drift helpers referenced by the architecture document.

This module includes small, deterministic example evolvers that can be
attached to Layer instances. They are intentionally simple and deterministic
so they are easy to test and reason about; complex domain logic should be
implemented by the user following the architecture PDF.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .qubit import ComputationalQubit

# Type alias for evolvers
EvolverFunc = Callable[[ComputationalQubit, float], ComputationalQubit]

_REGISTRY: dict[str, EvolverFunc] = {}


def register_evolver(name: str, fn: EvolverFunc) -> None:
    """Register an evolver function under a specific name.

    Parameters
    ----------
    name : str
        The name to register the evolver under.
    fn : EvolverFunc
        The evolver function to register.
    """
    _REGISTRY[name] = fn


def get_evolver(name: str) -> EvolverFunc:
    """Retrieve a registered evolver function by name.

    Parameters
    ----------
    name : str
        The name of the registered evolver.

    Returns
    -------
    EvolverFunc
        The requested evolver function.

    Raises
    ------
    KeyError
        If the evolver name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Evolver '{name}' not found in registry.")
    return _REGISTRY[name]


def list_evolvers() -> list[str]:
    """List all registered evolver names.

    Returns
    -------
    list[str]
        A list of registered evolver names.
    """
    return list(_REGISTRY.keys())


def linear_drift_evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
    """A tiny example evolver that shifts the drift vector by a small value.

    This is a placeholder: replace with the architecture's F_i mathematics.

    Parameters
    ----------
    qubit : ComputationalQubit
        The input qubit state.
    t : float
        The current time step.

    Returns
    -------
    ComputationalQubit
        The evolved qubit state.
    """
    q = qubit.copy()
    q.drift = q.drift + 0.001 * np.ones_like(q.drift)
    return q


def weighted_mix_evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
    """Mix primary and alternative using weights to produce a new primary.

    This demonstrates how weights could guide evolution.

    Parameters
    ----------
    qubit : ComputationalQubit
        The input qubit state.
    t : float
        The current time step.

    Returns
    -------
    ComputationalQubit
        The evolved qubit state.
    """
    q = qubit.copy()
    alpha, beta, _gamma = q.weights
    # simple convex combination in real space (for illustration only)
    p = (np.real(alpha) * q.primary + np.real(beta) * q.alternative) / max(1e-12, (np.real(alpha) + np.real(beta)))
    q.primary = p
    # small diffusion to drift
    q.drift = q.drift + 0.0005 * (np.random.default_rng(int(t)).normal(size=q.drift.shape))
    return q


def identity_evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
    """Returns a copy of the qubit unchanged.

    Parameters
    ----------
    qubit : ComputationalQubit
        The input qubit state.
    t : float
        The current time step.

    Returns
    -------
    ComputationalQubit
        The identical copied qubit state.
    """
    return qubit.copy()


def sinusoidal_evolver(amplitude: float = 0.01, frequency: float = 1.0) -> EvolverFunc:
    """Create an evolver that applies sinusoidal oscillation to the drift vector.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude of the oscillation, by default 0.01.
    frequency : float, optional
        The frequency of the oscillation, by default 1.0.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        q.drift = q.drift + amplitude * np.sin(2 * np.pi * frequency * t) * np.ones_like(q.drift)
        return q
    return _evolver


def exponential_decay_evolver(decay_rate: float = 0.01) -> EvolverFunc:
    """Create an evolver that decays the drift vector exponentially over time.

    Parameters
    ----------
    decay_rate : float, optional
        The decay rate, by default 0.01.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        q.drift = q.drift * np.exp(-decay_rate * t)
        return q
    return _evolver


def ou_evolver(theta: float = 0.1, mu: float = 0.0, sigma: float = 0.01, seed: int | None = None) -> EvolverFunc:
    """Create an Ornstein-Uhlenbeck (mean-reverting) noise evolver for the drift vector.

    Parameters
    ----------
    theta : float, optional
        The rate of mean reversion, by default 0.1.
    mu : float, optional
        The long-term mean, by default 0.0.
    sigma : float, optional
        The volatility/noise magnitude, by default 0.01.
    seed : int | None, optional
        Random seed, by default None.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    rng = np.random.default_rng(seed)

    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        dt = 1.0  # assumed constant timestep step
        noise = rng.normal(size=q.drift.shape)
        q.drift = q.drift + theta * (mu - q.drift) * dt + sigma * np.sqrt(dt) * noise
        return q
    return _evolver


def crossover_evolver(crossover_rate: float = 0.5, seed: int | None = None) -> EvolverFunc:
    """Create an evolver that performs genetic crossover between primary and alternative components.

    Parameters
    ----------
    crossover_rate : float, optional
        The probability of swapping elements between primary and alternative, by default 0.5.
    seed : int | None, optional
        Random seed, by default None.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    rng = np.random.default_rng(seed)

    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        mask = rng.random(size=q.primary.shape) < crossover_rate
        # Swap components where mask is True
        temp = q.primary[mask]
        q.primary[mask] = q.alternative[mask]
        q.alternative[mask] = temp
        return q
    return _evolver


def mutation_evolver(rate: float = 0.1, magnitude: float = 0.01, seed: int | None = None) -> EvolverFunc:
    """Create an evolver that applies random mutations to the primary component.

    Parameters
    ----------
    rate : float, optional
        The probability of mutating any given element, by default 0.1.
    magnitude : float, optional
        The scale of the mutation noise, by default 0.01.
    seed : int | None, optional
        Random seed, by default None.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    rng = np.random.default_rng(seed)

    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        mask = rng.random(size=q.primary.shape) < rate
        noise = rng.normal(scale=magnitude, size=q.primary.shape) + 1j * rng.normal(scale=magnitude, size=q.primary.shape)
        q.primary[mask] += noise[mask]
        return q
    return _evolver


def adaptive_evolver(base_step: float = 0.001, decay: float = 0.999) -> EvolverFunc:
    """Create an evolver where the step size decays exponentially with time.

    Parameters
    ----------
    base_step : float, optional
        The initial step size at t=0, by default 0.001.
    decay : float, optional
        The decay multiplier applied per unit of time, by default 0.999.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        step = base_step * (decay ** t)
        q.drift = q.drift + step * np.ones_like(q.drift)
        return q
    return _evolver


def composite_evolver(*evolvers: EvolverFunc) -> EvolverFunc:
    """Create an evolver that chains multiple evolvers in sequence.

    Parameters
    ----------
    evolvers : tuple[EvolverFunc, ...]
        A sequence of evolver functions to apply in order.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        for ev in evolvers:
            q = ev(q, t)
        return q
    return _evolver


def probabilistic_evolver(evolvers: Sequence[EvolverFunc], probabilities: Sequence[float], seed: int | None = None) -> EvolverFunc:
    """Create an evolver that randomly selects one evolver from a list to apply.

    Parameters
    ----------
    evolvers : Sequence[EvolverFunc]
        A list of evolver functions.
    probabilities : Sequence[float]
        The probability of selecting each corresponding evolver.
    seed : int | None, optional
        Random seed, by default None.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    rng = np.random.default_rng(seed)
    
    # Ensure probabilities are normalized
    probs = np.array(probabilities)
    probs = probs / np.sum(probs)

    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        # Choose index based on probabilities
        idx = rng.choice(len(evolvers), p=probs)
        return evolvers[idx](qubit, t)
    return _evolver


def gaussian_noise_evolver(sigma: float = 0.01, seed: int | None = None) -> EvolverFunc:
    """Create an evolver that adds complex Gaussian noise to the primary state.

    Parameters
    ----------
    sigma : float, optional
        The standard deviation of the Gaussian noise, by default 0.01.
    seed : int | None, optional
        Random seed, by default None.

    Returns
    -------
    EvolverFunc
        An evolver function.
    """
    rng = np.random.default_rng(seed)

    def _evolver(qubit: ComputationalQubit, t: float) -> ComputationalQubit:
        q = qubit.copy()
        noise_real = rng.normal(scale=sigma, size=q.primary.shape)
        noise_imag = rng.normal(scale=sigma, size=q.primary.shape)
        q.primary = q.primary + noise_real + 1j * noise_imag
        return q
    return _evolver


def make_evolver(config: dict[str, Any]) -> EvolverFunc:
    """Create an evolver from a configuration dictionary.

    The dictionary must contain a 'type' key specifying the evolver name,
    which corresponds to a registered evolver factory or function.
    Any other keys are passed as keyword arguments to the factory.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary, e.g., {'type': 'ou', 'theta': 0.1}.

    Returns
    -------
    EvolverFunc
        The instantiated evolver function.

    Raises
    ------
    ValueError
        If 'type' is not in the config.
    """
    if 'type' not in config:
        raise ValueError("Config must contain a 'type' key specifying the evolver.")
    
    cfg = config.copy()
    evolver_type = cfg.pop('type')
    
    # Pre-registered standard factories
    factories: dict[str, Any] = {
        'sinusoidal': sinusoidal_evolver,
        'exponential_decay': exponential_decay_evolver,
        'ou': ou_evolver,
        'crossover': crossover_evolver,
        'mutation': mutation_evolver,
        'adaptive': adaptive_evolver,
        'gaussian_noise': gaussian_noise_evolver,
    }
    
    if evolver_type in factories:
        evolver_fn = factories[evolver_type](**cfg)
        return evolver_fn
    elif evolver_type in _REGISTRY:
        # A direct evolver function (not a factory) in registry
        # We assume it doesn't take factory args unless we register factories there.
        # It's safer to just return the registered func.
        return _REGISTRY[evolver_type]
    else:
        raise ValueError(f"Unknown evolver type '{evolver_type}' in config.")


# Register default evolvers
register_evolver('linear_drift', linear_drift_evolver)
register_evolver('weighted_mix', weighted_mix_evolver)
register_evolver('identity', identity_evolver)
