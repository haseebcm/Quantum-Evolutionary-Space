"""Coordinator for the QSEE-11L eleven-layer construct."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .layer import Layer
from .qubit import ComputationalQubit


@dataclass
class QSEE11L:
    """Top-level coordinator that holds independent Layer objects.

    It provides convenience helpers for stepping, snapshots, aggregation,
    and evolutionary strategies across layers.
    """

    layers: list[Layer] = field(default_factory=lambda: [Layer(name=f"layer_{i+1}") for i in range(11)])
    groups: dict[str, list[int]] = field(default_factory=dict)
    step_count: int = 0

    def step_all(self, t: float = 0.0) -> None:
        """Advance all layers by one step. Layers are independent and stepped in order."""
        for layer in self.layers:
            layer.step(t)
        self.step_count += 1

    def snapshot(self) -> list[dict]:
        return [layer.snapshot() for layer in self.layers]

    def characteristic_matrix(self) -> np.ndarray:
        """Return a stacked array of per-layer characteristic vectors (real-valued)."""
        vectors = [layer.qubit.as_vector() for layer in self.layers]
        return np.vstack(vectors)

    def set_evolver_for_layer(self, index: int, fn: Callable) -> None:
        self.layers[index].set_evolver(fn)

    def aggregate_weights(self) -> np.ndarray:
        """Return per-layer weight arrays stacked into shape (N, 3)."""
        return np.vstack([layer.qubit.weights for layer in self.layers])

    def step_layers(self, indices: list[int], t: float = 0.0) -> None:
        """Advance specific layers by one step."""
        for idx in indices:
            self.layers[idx].step(t)
        self.step_count += 1

    def create_group(self, name: str, indices: list[int]) -> None:
        """Define a named group of layers by their indices."""
        self.groups[name] = indices

    def step_group(self, name: str, t: float = 0.0) -> None:
        """Advance a specific group of layers by one step."""
        if name in self.groups:
            self.step_layers(self.groups[name], t)

    def synchronize(self, strategy: str = "average") -> None:
        """Synchronize the states (weights) of all layers based on a strategy.
        
        Args:
            strategy: 'average', 'median', or 'best'
        """
        if not self.layers:
            return

        if strategy == "average":
            avg_weights = np.mean([layer.qubit.weights for layer in self.layers], axis=0)
            for layer in self.layers:
                layer.qubit.weights = avg_weights.copy()
        elif strategy == "median":
            med_weights = np.median([layer.qubit.weights for layer in self.layers], axis=0)
            for layer in self.layers:
                layer.qubit.weights = med_weights.copy()
        elif strategy == "best":
            _, best_lay = self.best_layer()
            best_weights = best_lay.qubit.weights.copy()
            for layer in self.layers:
                layer.qubit.weights = best_weights.copy()
        else:
            raise ValueError(f"Unknown synchronization strategy: {strategy}")

    def set_fitness_fn(self, fn: Callable[[ComputationalQubit], float]) -> None:
        """Set the fitness function for all layers."""
        for layer in self.layers:
            layer.set_fitness_fn(fn)

    def fitness_matrix(self) -> np.ndarray:
        """Return the fitness history of all layers as a 2D array."""
        if not self.layers:
            return np.array([])
            
        max_len = max((len(layer.fitness_history) for layer in self.layers), default=0)
        matrix = np.zeros((len(self.layers), max_len))
        for i, layer in enumerate(self.layers):
            hist_len = len(layer.fitness_history)
            if hist_len > 0:
                matrix[i, :hist_len] = layer.fitness_history
                if hist_len < max_len:
                    matrix[i, hist_len:] = layer.fitness_history[-1]
        return matrix

    def best_layer(self) -> tuple[int, Layer]:
        """Return the index and layer with the highest current fitness."""
        if not self.layers:
            raise ValueError("No layers to evaluate.")
        best_idx = max(range(len(self.layers)), key=lambda i: self.layers[i].fitness)
        return best_idx, self.layers[best_idx]

    def worst_layer(self) -> tuple[int, Layer]:
        """Return the index and layer with the lowest current fitness."""
        if not self.layers:
            raise ValueError("No layers to evaluate.")
        worst_idx = min(range(len(self.layers)), key=lambda i: self.layers[i].fitness)
        return worst_idx, self.layers[worst_idx]

    def diversity(self) -> float:
        """Calculate the mean pairwise L2 distance between the states of all layers."""
        if len(self.layers) < 2:
            return 0.0
        vectors = np.array([layer.qubit.as_vector() for layer in self.layers])
        diffs = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=-1)
        # Sum of upper triangle
        total_distance = np.sum(np.triu(distances))
        num_pairs = len(self.layers) * (len(self.layers) - 1) / 2
        return float(total_distance / num_pairs) if num_pairs > 0 else 0.0

    def replace_layer(self, index: int, new_layer: Layer) -> None:
        """Replace a layer at the given index."""
        self.layers[index] = new_layer

    def run(self, steps: int, t_start: float = 0.0, dt: float = 1.0) -> list[list[dict]]:
        """Run all layers for multiple steps and return their snapshots per step."""
        results = []
        for i in range(steps):
            self.step_all(t_start + i * dt)
            results.append(self.snapshot())
        return results

    def evolve_competitive(self, rounds: int = 1, cull_fraction: float = 0.3, t: float = 0.0) -> None:
        """Replace the worst performing layers with mutations of the best.
        
        Args:
            rounds: Number of evolutionary rounds.
            cull_fraction: Fraction of layers to replace per round.
            t: Time parameter for steps.
        """
        if not self.layers:
            return
            
        num_cull = max(1, int(len(self.layers) * cull_fraction))
        for _ in range(rounds):
            self.step_all(t)
            
            # Sort indices by fitness descending
            sorted_indices = sorted(range(len(self.layers)), key=lambda i: self.layers[i].fitness, reverse=True)
            best_idx = sorted_indices[0]
            worst_indices = sorted_indices[-num_cull:]
            
            best_layer = self.layers[best_idx]
            for idx in worst_indices:
                # Create a mutated copy of the best layer
                new_layer = Layer(name=f"{best_layer.name}_mutated_{self.step_count}")
                new_layer.qubit.weights = best_layer.qubit.weights.copy()
                # Apply slight mutation
                new_layer.qubit.weights += np.random.normal(0, 0.1, size=new_layer.qubit.weights.shape)
                new_layer.evolve_fn = best_layer.evolve_fn
                new_layer.evolve_fns = best_layer.evolve_fns.copy()
                new_layer.fitness_fn = best_layer.fitness_fn
                self.replace_layer(idx, new_layer)

    def evolve_cooperative(self, rounds: int = 1, t: float = 0.0) -> None:
        """Evolve layers cooperatively, periodically sharing knowledge via synchronization."""
        for _ in range(rounds):
            self.step_all(t)
            self.synchronize(strategy="average")

    def convergence_report(self) -> dict:
        """Return a dictionary of convergence status for each layer."""
        return {layer.name: layer.has_converged() for layer in self.layers}

    def telemetry(self) -> dict:
        """Return comprehensive telemetry metrics for the entire coordinator."""
        return {
            "step_count": self.step_count,
            "num_layers": len(self.layers),
            "diversity": self.diversity(),
            "best_fitness": self.layers[self.best_layer()[0]].fitness if self.layers else 0.0,
            "worst_fitness": self.layers[self.worst_layer()[0]].fitness if self.layers else 0.0,
            "convergence": self.convergence_report(),
        }

    def reset(self) -> None:
        """Reset all layers and the coordinator state."""
        for layer in self.layers:
            layer.reset()
        self.step_count = 0

    def add_layer(self, layer: Layer) -> None:
        """Append a new layer to the coordinator."""
        self.layers.append(layer)

    def remove_layer(self, index: int) -> Layer:
        """Remove and return the layer at the given index."""
        return self.layers.pop(index)
