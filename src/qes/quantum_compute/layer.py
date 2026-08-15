"""Per-layer manager for one QSEE-11L evolutionary layer."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .qubit import ComputationalQubit


@dataclass
class Layer:
    """Manager for a single independent evolution layer.

    The Layer wraps a ComputationalQubit and exposes a pluggable evolution
    function F(state, t) -> state_delta. The evolution function should be
    deterministic and not access other layers.
    """

    name: str
    qubit: ComputationalQubit = field(default_factory=ComputationalQubit)
    evolve_fn: Callable[[ComputationalQubit, float], ComputationalQubit] | None = None
    
    # Enhanced tracking fields
    history: list[dict] = field(default_factory=list)
    fitness_fn: Callable[[ComputationalQubit], float] | None = None
    fitness_history: list[float] = field(default_factory=list)
    on_step_callbacks: list[Callable[[Layer, float], None]] = field(default_factory=list)
    evolve_fns: list[Callable[[ComputationalQubit, float], ComputationalQubit]] = field(default_factory=list)
    step_count: int = 0

    def step(self, t: float = 0.0) -> None:
        """Apply one evolution step using the layer's evolve_fns or evolve_fn if present."""
        fns = self.evolve_fns if self.evolve_fns else ([self.evolve_fn] if self.evolve_fn else [])
        for fn in fns:
            result = fn(self.qubit, t)
            if isinstance(result, ComputationalQubit):
                self.qubit = result
                
        self.step_count += 1
        self.history.append(self.snapshot())
        
        if self.fitness_fn is not None:
            self.fitness_history.append(self.fitness_fn(self.qubit))
            
        for cb in self.on_step_callbacks:
            cb(self, t)

    def set_evolver(self, fn: Callable[[ComputationalQubit, float], ComputationalQubit]) -> None:
        self.evolve_fn = fn

    def snapshot(self) -> dict:
        return {"name": self.name, "vector": self.qubit.as_vector(), "weights": self.qubit.weights.copy()}

    def get_history(self) -> list[dict]:
        """Return a copy of the layer's snapshot history."""
        return self.history.copy()

    def run(self, steps: int, t_start: float = 0.0, dt: float = 1.0) -> list[dict]:
        """Evolve multiple steps, returning the per-step snapshots."""
        snapshots = []
        for i in range(steps):
            self.step(t_start + i * dt)
            snapshots.append(self.history[-1])
        return snapshots

    @property
    def fitness(self) -> float:
        """Return the current fitness score, or 0.0 if no fitness function is set."""
        if self.fitness_fn is None:
            return 0.0
        return self.fitness_fn(self.qubit)

    def set_fitness_fn(self, fn: Callable[[ComputationalQubit], float]) -> None:
        """Set the fitness function."""
        self.fitness_fn = fn

    def has_converged(self, atol: float = 1e-6, window: int = 5) -> bool:
        """Check if recent state changes are below atol over the given window."""
        if len(self.history) < window + 1:
            return False
        recent = self.history[-(window + 1):]
        for i in range(1, len(recent)):
            prev_vec = np.array(recent[i-1]["vector"])
            curr_vec = np.array(recent[i]["vector"])
            if not np.allclose(prev_vec, curr_vec, atol=atol):
                return False
        return True

    def checkpoint(self) -> dict:
        """Create a full checkpoint of the layer's state."""
        return {
            "name": self.name,
            "qubit_weights": self.qubit.weights.copy() if hasattr(self.qubit, "weights") and self.qubit.weights is not None else None,
            "step_count": self.step_count,
            "history": self.history.copy(),
            "fitness_history": self.fitness_history.copy(),
        }

    def restore(self, checkpoint: dict) -> None:
        """Restore the layer's state from a checkpoint."""
        self.name = checkpoint["name"]
        if checkpoint["qubit_weights"] is not None:
            self.qubit.weights = np.array(checkpoint["qubit_weights"])
        self.step_count = checkpoint["step_count"]
        self.history = checkpoint["history"].copy()
        self.fitness_history = checkpoint["fitness_history"].copy()

    def stats(self) -> dict:
        """Return statistics about the layer's evolution."""
        drift_magnitude = 0.0
        if len(self.history) >= 2:
            prev_v = np.array(self.history[-2]["vector"])
            curr_v = np.array(self.history[-1]["vector"])
            drift_magnitude = float(np.linalg.norm(curr_v - prev_v))
        
        weight_entropy = 0.0
        if hasattr(self.qubit, "weights") and self.qubit.weights is not None:
            w = np.abs(self.qubit.weights)
            w_sum = np.sum(w)
            if w_sum > 0:
                p = w / w_sum
                # filter 0 values to avoid log2 issues
                p_nz = p[p > 0]
                weight_entropy = float(-np.sum(p_nz * np.log2(p_nz)))
            
        fitness_trend = 0.0
        if len(self.fitness_history) >= 2:
            fitness_trend = self.fitness_history[-1] - self.fitness_history[-2]

        return {
            "step_count": self.step_count,
            "drift_magnitude": drift_magnitude,
            "weight_entropy": weight_entropy,
            "fitness_trend": fitness_trend,
        }

    def on_step(self, callback: Callable[[Layer, float], None]) -> None:
        """Add a callback to be executed after each step."""
        self.on_step_callbacks.append(callback)

    def add_evolver(self, fn: Callable[[ComputationalQubit, float], ComputationalQubit]) -> None:
        """Add an evolution function to the chain."""
        self.evolve_fns.append(fn)

    def reset(self) -> None:
        """Reset the qubit to default and clear history."""
        self.qubit = ComputationalQubit()
        self.history.clear()
        self.fitness_history.clear()
        self.step_count = 0
        self.evolve_fns.clear()
        self.on_step_callbacks.clear()
