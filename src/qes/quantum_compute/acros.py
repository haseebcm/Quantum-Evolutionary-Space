"""ACROS-style correction helpers.

These functions implement the high-level correction gradient motif described in
the architecture: a configurable correction applied to a numeric state vector
based on a penalty/phi and optionally a CCI term. Implemented as a tiny,
well-documented helper to integrate into Layer evolvers or higher-level
controllers.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def acros_correction(x: np.ndarray, grad_phi: np.ndarray, K: float, mu: float = 0.0, cci: float = 0.0) -> np.ndarray:
    """Apply a conservative ACROS correction step to state vector x.

    x <- x - K * (grad_phi + mu * cci)

    Notes
    -----
    - grad_phi should be the numeric gradient of the target potential Phi(x).
    - cci is a scalar cascade/collapse index; mu scales its influence.
    - This is a deterministic, software-only approximation for integration.
    """
    step = K * (np.asarray(grad_phi, dtype=float) + mu * float(cci))
    return np.asarray(x, dtype=float) - step


def make_gradient_estimator(f: Callable[[np.ndarray], float], eps: float = 1e-6) -> Callable[[np.ndarray], np.ndarray]:
    """Return a finite-difference gradient estimator for f.

    Simple central-difference estimator used for small-dimensional vectors.
    """

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        for i in range(x.size):
            dx = np.zeros_like(x)
            dx.ravel()[i] = eps
            g.ravel()[i] = (f(x + dx) - f(x - dx)) / (2.0 * eps)
        return g

    return grad


@dataclass
class CorrectionResult:
    """Dataclass holding the result of multi-step correction."""
    x: np.ndarray
    converged: bool
    steps_taken: int
    final_phi: float
    trajectory: list[np.ndarray]
    phi_history: list[float]


def acros_multi_step(
    x: np.ndarray,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    K: float,
    max_steps: int = 100,
    atol: float = 1e-8,
    mu: float = 0.0,
    cci: float = 0.0,
) -> CorrectionResult:
    """Apply multi-step ACROS correction until convergence or max_steps.

    Parameters
    ----------
    x : np.ndarray
        Initial state vector.
    f : Callable[[np.ndarray], float]
        Potential function.
    grad_f : Callable[[np.ndarray], np.ndarray]
        Gradient of the potential function.
    K : float
        Step size.
    max_steps : int, optional
        Maximum number of steps, by default 100.
    atol : float, optional
        Absolute tolerance on gradient norm for convergence, by default 1e-8.
    mu : float, optional
        Scale factor for CCI, by default 0.0.
    cci : float, optional
        Cascade/collapse index, by default 0.0.

    Returns
    -------
    CorrectionResult
        The result of the optimization.
    """
    x_curr = np.asarray(x, dtype=float).copy()
    trajectory = [x_curr.copy()]
    phi_history = [f(x_curr)]
    converged = False

    for _ in range(max_steps):
        g = grad_f(x_curr)
        if gradient_norm(g) < atol:
            converged = True
            break
            
        x_curr = acros_correction(x_curr, g, K, mu, cci)
        trajectory.append(x_curr.copy())
        phi_history.append(f(x_curr))

    return CorrectionResult(
        x=x_curr,
        converged=converged,
        steps_taken=len(trajectory) - 1,
        final_phi=phi_history[-1],
        trajectory=trajectory,
        phi_history=phi_history,
    )


def acros_adaptive(
    x: np.ndarray,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    K_init: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_steps: int = 100,
    atol: float = 1e-8,
) -> CorrectionResult:
    """Apply ACROS correction with Armijo backtracking line search.

    Parameters
    ----------
    x : np.ndarray
        Initial state vector.
    f : Callable[[np.ndarray], float]
        Potential function.
    grad_f : Callable[[np.ndarray], np.ndarray]
        Gradient of the potential function.
    K_init : float, optional
        Initial step size, by default 1.0.
    c : float, optional
        Sufficient decrease constant, by default 1e-4.
    rho : float, optional
        Backtracking decay factor, by default 0.5.
    max_steps : int, optional
        Maximum number of outer steps, by default 100.
    atol : float, optional
        Absolute tolerance on gradient norm for convergence, by default 1e-8.

    Returns
    -------
    CorrectionResult
        The result of the optimization.
    """
    x_curr = np.asarray(x, dtype=float).copy()
    trajectory = [x_curr.copy()]
    phi_history = [f(x_curr)]
    converged = False

    for _ in range(max_steps):
        g = grad_f(x_curr)
        if gradient_norm(g) < atol:
            converged = True
            break

        f_curr = phi_history[-1]
        p = -g  # Descent direction
        m = np.dot(g.ravel(), p.ravel())

        K = K_init
        x_next = x_curr + K * p
        while f(x_next) > f_curr + c * K * m:
            K *= rho
            x_next = x_curr + K * p
            if K < 1e-14:
                break

        x_curr = x_next
        trajectory.append(x_curr.copy())
        phi_history.append(f(x_curr))

    return CorrectionResult(
        x=x_curr,
        converged=converged,
        steps_taken=len(trajectory) - 1,
        final_phi=phi_history[-1],
        trajectory=trajectory,
        phi_history=phi_history,
    )


def acros_momentum(
    x: np.ndarray,
    grad_phi: np.ndarray,
    K: float,
    velocity: np.ndarray,
    beta: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ACROS correction with momentum.

    Parameters
    ----------
    x : np.ndarray
        State vector.
    grad_phi : np.ndarray
        Gradient of the potential function.
    K : float
        Step size (learning rate).
    velocity : np.ndarray
        Current momentum velocity vector.
    beta : float, optional
        Momentum decay factor, by default 0.9.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The updated state vector and the updated velocity vector.
    """
    new_velocity = beta * velocity + (1 - beta) * grad_phi
    new_x = np.asarray(x, dtype=float) - K * new_velocity
    return new_x, new_velocity


def acros_adam(
    x: np.ndarray,
    grad_phi: np.ndarray,
    K: float,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Adam-style optimization step.

    Parameters
    ----------
    x : np.ndarray
        State vector.
    grad_phi : np.ndarray
        Gradient of the potential function.
    K : float
        Step size (learning rate).
    m : np.ndarray
        First moment estimate.
    v : np.ndarray
        Second moment estimate.
    t : int
        Current timestep.
    beta1 : float, optional
        First moment decay rate, by default 0.9.
    beta2 : float, optional
        Second moment decay rate, by default 0.999.
    eps : float, optional
        Small term to prevent division by zero, by default 1e-8.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The updated state vector, updated first moment, and updated second moment.
    """
    new_m = beta1 * m + (1 - beta1) * grad_phi
    new_v = beta2 * v + (1 - beta2) * (grad_phi**2)
    m_hat = new_m / (1 - beta1**t)
    v_hat = new_v / (1 - beta2**t)
    new_x = np.asarray(x, dtype=float) - K * m_hat / (np.sqrt(v_hat) + eps)
    return new_x, new_m, new_v


def projected_acros(
    x: np.ndarray,
    grad_phi: np.ndarray,
    K: float,
    bounds_low: np.ndarray | float,
    bounds_high: np.ndarray | float,
    mu: float = 0.0,
    cci: float = 0.0,
) -> np.ndarray:
    """Apply ACROS correction and project the result back into feasible region.

    Parameters
    ----------
    x : np.ndarray
        State vector.
    grad_phi : np.ndarray
        Gradient of the potential function.
    K : float
        Step size.
    bounds_low : np.ndarray | float
        Lower bounds for the projection.
    bounds_high : np.ndarray | float
        Upper bounds for the projection.
    mu : float, optional
        Scale factor for CCI, by default 0.0.
    cci : float, optional
        Cascade/collapse index, by default 0.0.

    Returns
    -------
    np.ndarray
        The projected, corrected state vector.
    """
    new_x = acros_correction(x, grad_phi, K, mu, cci)
    return np.clip(new_x, bounds_low, bounds_high)


def make_batch_gradient_estimator(
    f: Callable[[np.ndarray], float],
    batch_size: int = 5,
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a stochastic gradient estimator sampling random coordinates.

    Parameters
    ----------
    f : Callable[[np.ndarray], float]
        Potential function.
    batch_size : int, optional
        Number of coordinates to estimate per call, by default 5.
    eps : float, optional
        Finite difference step, by default 1e-6.
    rng : np.random.Generator | None, optional
        Random number generator for reproducible sampling.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Gradient estimator function.
    """
    if rng is None:
        rng = np.random.default_rng()

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        size = x.size
        sample_size = min(batch_size, size)
        indices = rng.choice(size, size=sample_size, replace=False)
        
        flat_g = g.ravel()
        for i in indices:
            dx = np.zeros_like(x)
            dx.ravel()[i] = eps
            flat_g[i] = (f(x + dx) - f(x - dx)) / (2.0 * eps)
        return g

    return grad


def make_forward_gradient_estimator(
    f: Callable[[np.ndarray], float], eps: float = 1e-6
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a forward-difference gradient estimator for f.

    Cheaper but less accurate than central-difference.

    Parameters
    ----------
    f : Callable[[np.ndarray], float]
        Potential function.
    eps : float, optional
        Finite difference step, by default 1e-6.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Gradient estimator function.
    """

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        f_val = f(x)
        for i in range(x.size):
            dx = np.zeros_like(x)
            dx.ravel()[i] = eps
            g.ravel()[i] = (f(x + dx) - f_val) / eps
        return g

    return grad


def gradient_norm(grad: np.ndarray) -> float:
    """Compute the L2 norm of the gradient.

    Parameters
    ----------
    grad : np.ndarray
        Gradient vector.

    Returns
    -------
    float
        The L2 norm of the gradient.
    """
    return float(np.linalg.norm(grad))


def cosine_decay(K_init: float, step: int, total_steps: int) -> float:
    """Compute step size with cosine decay schedule.

    Parameters
    ----------
    K_init : float
        Initial step size.
    step : int
        Current step number.
    total_steps : int
        Total number of steps in the schedule.

    Returns
    -------
    float
        The decayed step size.
    """
    if step >= total_steps:
        return 0.0
    return K_init * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))


def exponential_decay(K_init: float, step: int, decay_rate: float) -> float:
    """Compute step size with exponential decay schedule.

    Parameters
    ----------
    K_init : float
        Initial step size.
    step : int
        Current step number.
    decay_rate : float
        Decay rate.

    Returns
    -------
    float
        The decayed step size.
    """
    return K_init * (decay_rate**step)


def step_decay(K_init: float, step: int, drop_every: int, drop_factor: float) -> float:
    """Compute step size with step decay schedule.

    Parameters
    ----------
    K_init : float
        Initial step size.
    step : int
        Current step number.
    drop_every : int
        Number of steps before each drop.
    drop_factor : float
        Factor to multiply step size by at each drop.

    Returns
    -------
    float
        The decayed step size.
    """
    num_drops = step // drop_every
    return K_init * (drop_factor**num_drops)
