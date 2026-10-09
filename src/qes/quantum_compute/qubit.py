"""Computational qubit abstraction.

This module implements a light-weight, software-only ComputationalQubit that
captures the N, N', and drift states plus normalized complex weights as
specified in the architecture document. It intentionally avoids claiming any
physical quantum behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComputationalQubit:
    """A software-only, quantum-inspired computational qubit.

    Attributes
    ----------
    primary : np.ndarray
        Primary (N) state vector.
    alternative : np.ndarray
        Alternative (N') state vector.
    drift : np.ndarray
        Drift/evolution state vector.
    weights : np.ndarray
        Complex-valued weights [alpha, beta, gamma] that should be normalized.
    """

    primary: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    alternative: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    drift: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=complex))

    def _quantum_state(self) -> np.ndarray:
        """Normalize a finite vector for physical-state diagnostics.

        primary remains an arbitrary computational vector; physical methods
        operate on its normalized ray and reject zero/invalid vectors.
        """
        state = np.asarray(self.primary, dtype=complex)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("primary must be a finite nonempty state vector")
        scale = float(np.max(np.abs(state)))
        if scale == 0:
            raise ValueError("zero state has no normalized quantum representation")
        scaled = state / scale
        return scaled / np.linalg.norm(scaled)

    def normalize_weights(self) -> None:
        """Normalize weights so that sum(|w|^2) == 1."""
        magsq = np.sum(np.abs(self.weights) ** 2)
        if magsq == 0:
            raise ValueError("weights have zero magnitude and cannot be normalized")
        self.weights = self.weights / np.sqrt(magsq)

    def as_vector(self) -> np.ndarray:
        """Return a real-valued concatenated representation for downstream use."""
        return np.concatenate([np.asarray(self.primary, dtype=float), np.asarray(self.alternative, dtype=float), np.asarray(self.drift, dtype=float)])

    def copy(self) -> ComputationalQubit:
        return ComputationalQubit(self.primary.copy(), self.alternative.copy(), self.drift.copy(), self.weights.copy())

    def norm(self) -> float:
        """Calculate the L2 norm of the primary state.
        
        Returns
        -------
        float
            Norm of the primary state.
        """
        return float(np.linalg.norm(self.primary))

    def inner(self, other: ComputationalQubit) -> complex:
        """Calculate the inner product with another qubit.
        
        Parameters
        ----------
        other : ComputationalQubit
            The other qubit.
            
        Returns
        -------
        complex
            The inner product <self|other>.
        """
        return complex(np.vdot(self.primary, other.primary))

    def __eq__(self, other: object) -> bool:
        """Check equality based on primary, alternative, and drift vectors."""
        if not isinstance(other, ComputationalQubit):
            return NotImplemented
        return (
            np.allclose(self.primary, other.primary)
            and np.allclose(self.alternative, other.alternative)
            and np.allclose(self.drift, other.drift)
            and np.allclose(self.weights, other.weights)
        )

    def __repr__(self) -> str:
        """Return string representation using Dirac notation for simple basis states."""
        if self.primary.size == 2 and np.allclose(self.primary, [1, 0]):
            state = "|0⟩"
        elif self.primary.size == 2 and np.allclose(self.primary, [0, 1]):
            state = "|1⟩"
        else:
            state = str(self.primary)
        return f"ComputationalQubit({state})"

    def density_matrix(self) -> np.ndarray:
        """Calculate the density matrix from the primary state.
        
        Returns
        -------
        np.ndarray
            The density matrix (outer product of the primary state).
        """
        state = self._quantum_state()
        return np.outer(state, state.conj())

    def to_bloch(self) -> tuple[float, float]:
        """Get the Bloch sphere coordinates (theta, phi) for a 2D primary state.
        
        Returns
        -------
        tuple[float, float]
            Theta and phi angles in radians.
        """
        if self.primary.size != 2:
            raise ValueError("Bloch sphere representation is only valid for 2D states.")
        
        # Ensure state is normalized for Bloch sphere mapping
        state = self._quantum_state()
        
        alpha, beta = state
        theta = 2 * np.arccos(np.clip(np.abs(alpha), 0, 1))
        phi = np.angle(beta) - np.angle(alpha)
        
        # Ensure phi is in [0, 2pi)
        if phi < 0:
            phi += 2 * np.pi
            
        return float(theta), float(phi)

    def purity(self) -> float:
        """Calculate the purity of the state based on its density matrix.
        
        Returns
        -------
        float
            Purity of the state (Trace(rho^2)).
        """
        rho = self.density_matrix()
        return float(np.real(np.trace(rho @ rho)))

    def von_neumann_entropy(self) -> float:
        """Calculate the Von Neumann entropy of the state.
        
        Returns
        -------
        float
            The entropy S(rho) = -Tr(rho * log2(rho)).
        """
        rho = self.density_matrix()
        # Find eigenvalues, filter out zero and negative values for numerical stability
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues))) if len(eigenvalues) > 0 else 0.0

    def fidelity(self, other: ComputationalQubit) -> float:
        """Calculate the fidelity between this and another qubit.
        
        Parameters
        ----------
        other : ComputationalQubit
            The other qubit.
            
        Returns
        -------
        float
            Fidelity |<self|other>|^2.
        """
        left, right = self._quantum_state(), other._quantum_state()
        if left.shape != right.shape:
            raise ValueError("state dimensions must match")
        return float(np.clip(np.abs(np.vdot(left, right)) ** 2, 0.0, 1.0))

    def trace_distance(self, other: ComputationalQubit) -> float:
        """Calculate the trace distance between this and another qubit.
        
        Parameters
        ----------
        other : ComputationalQubit
            The other qubit.
            
        Returns
        -------
        float
            Trace distance.
        """
        rho1 = self.density_matrix()
        rho2 = other.density_matrix()
        diff = rho1 - rho2
        
        # Trace distance = 0.5 * Tr(|rho1 - rho2|) where |A| = sqrt(A^dagger A)
        # For hermitian diff, it's 0.5 * sum of absolute eigenvalues
        eigvals = np.linalg.eigvalsh(diff)
        return float(0.5 * np.sum(np.abs(eigvals)))

    def bures_distance(self, other: ComputationalQubit) -> float:
        """Calculate the Bures distance between this and another qubit.
        
        Parameters
        ----------
        other : ComputationalQubit
            The other qubit.
            
        Returns
        -------
        float
            Bures distance.
        """
        f = self.fidelity(other)
        return float(np.sqrt(2 * (1 - np.sqrt(f))))

    @classmethod
    def from_density_matrix(cls, rho: np.ndarray) -> ComputationalQubit:
        """Perform state tomography by creating a qubit from a density matrix.
        
        This assumes the state is pure and extracts the eigenvector corresponding
        to the largest eigenvalue as the primary state.
        
        Parameters
        ----------
        rho : np.ndarray
            The density matrix.
            
        Returns
        -------
        ComputationalQubit
            A new qubit instance.
        """
        rho = np.asarray(rho, dtype=complex)
        if (rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.size == 0
                or not np.all(np.isfinite(rho)) or not np.allclose(rho, rho.conj().T)
                or not np.isclose(np.trace(rho), 1.0)):
            raise ValueError("rho must be a finite Hermitian trace-one density matrix")
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        if np.min(eigenvalues) < -1e-10 or not np.isclose(eigenvalues[-1], 1.0):
            raise ValueError("only pure positive density matrices are supported")
        # Get the eigenvector corresponding to the largest eigenvalue
        max_idx = np.argmax(eigenvalues)
        primary = eigenvectors[:, max_idx]
        return cls(primary=primary)

    @classmethod
    def superposition(cls, states: list[ComputationalQubit], amplitudes: list[complex]) -> ComputationalQubit:
        """Create a new qubit as a superposition of given states.
        
        Parameters
        ----------
        states : list[ComputationalQubit]
            List of basis states.
        amplitudes : list[complex]
            List of complex amplitudes.
            
        Returns
        -------
        ComputationalQubit
            A new qubit in the superposition state.
        """
        if len(states) != len(amplitudes):
            raise ValueError("Number of states must match number of amplitudes.")
        
        if not states:
            return cls()
            
        size = states[0].primary.size
        primary = np.zeros(size, dtype=complex)
        
        for state, amp in zip(states, amplitudes, strict=False):
            primary += amp * state.primary
            
        return cls(primary=primary)

    def measure(self, rng: np.random.Generator | None = None) -> int:
        """Measure the qubit in the computational basis.
        
        Collapses the primary state probabilistically to a basis state based on
        the Born rule probabilities.
        
        Parameters
        ----------
        rng : np.random.Generator | None, optional
            Random number generator for reproducibility.
            
        Returns
        -------
        int
            The index of the collapsed basis state.
        """
        if rng is None:
            rng = np.random.default_rng()
            
        # Calculate probabilities from primary state
        probs = np.abs(self._quantum_state()) ** 2
        
        # Sample an index based on probabilities
        index = int(rng.choice(len(probs), p=probs))
        
        # Collapse the state
        collapsed = np.zeros_like(self.primary)
        collapsed[index] = 1.0
        self.primary = collapsed
        
        return index

    def tensor(self, other: ComputationalQubit) -> ComputationalQubit:
        """Compute the tensor (Kronecker) product with another qubit.
        
        Parameters
        ----------
        other : ComputationalQubit
            The other qubit.
            
        Returns
        -------
        ComputationalQubit
            A new qubit representing the tensor product state.
        """
        primary = np.kron(self.primary, other.primary)
        alternative = np.kron(self.alternative, other.alternative)
        drift = np.kron(self.drift, other.drift)
        weights = np.kron(self.weights, other.weights)
        
        return ComputationalQubit(
            primary=primary,
            alternative=alternative,
            drift=drift,
            weights=weights
        )
