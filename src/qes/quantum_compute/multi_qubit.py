"""Multi-qubit linear-algebra helpers.

Provides tensor-product utilities and canonical basis vector generators. These
are pure Python / NumPy linear-algebra representations (mathematical models)
that mirror common quantum computing primitives. They are NOT tied to real
quantum hardware; they implement the algebraic model used by quantum
computing for reasoning and testing.
"""
from __future__ import annotations

import numpy as np


def kron(*matrices: np.ndarray) -> np.ndarray:
    """Compute the Kronecker product of the given matrices/vectors."""
    if not matrices:
        return np.array([1.0], dtype=complex)
    result = np.asarray(matrices[0], dtype=complex)
    for m in matrices[1:]:
        result = np.kron(result, np.asarray(m, dtype=complex))
    return result


def basis_state(index: int, num_qubits: int) -> np.ndarray:
    """Return the computational basis state |index> as a 2^n vector."""
    dim = 1 << num_qubits
    vec = np.zeros(dim, dtype=complex)
    vec[int(index) % dim] = 1.0
    return vec


def zero_state(num_qubits: int) -> np.ndarray:
    """Return the all-zero computational basis state |0...0>."""
    return basis_state(0, num_qubits)


def random_state(num_qubits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a normalized random complex state vector of 2^n dimension."""
    rng = rng or np.random.default_rng()
    dim = 1 << num_qubits
    re = rng.normal(size=dim)
    im = rng.normal(size=dim)
    v = re + 1j * im
    v /= np.linalg.norm(v)
    return v


def bell_state(index: int) -> np.ndarray:
    """
    Return one of the four Bell states.

    Parameters
    ----------
    index : int
        0 for |Φ+>, 1 for |Φ->, 2 for |Ψ+>, 3 for |Ψ->.

    Returns
    -------
    np.ndarray
        The Bell state vector.
    """
    vec = np.zeros(4, dtype=complex)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if index == 0:
        vec[0] = inv_sqrt2
        vec[3] = inv_sqrt2
    elif index == 1:
        vec[0] = inv_sqrt2
        vec[3] = -inv_sqrt2
    elif index == 2:
        vec[1] = inv_sqrt2
        vec[2] = inv_sqrt2
    elif index == 3:
        vec[1] = inv_sqrt2
        vec[2] = -inv_sqrt2
    else:
        raise ValueError("Bell state index must be 0, 1, 2, or 3.")
    return vec


def ghz_state(n: int) -> np.ndarray:
    """
    Return the n-qubit GHZ state: (|0...0> + |1...1>) / sqrt(2).

    Parameters
    ----------
    n : int
        The number of qubits.

    Returns
    -------
    np.ndarray
        The GHZ state vector.
    """
    if n < 1:
        raise ValueError("Number of qubits must be at least 1.")
    dim = 1 << n
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0 / np.sqrt(2.0)
    vec[-1] = 1.0 / np.sqrt(2.0)
    return vec


def w_state(n: int) -> np.ndarray:
    """
    Return the n-qubit W state, which is the equal superposition of all states with exactly one excitation.

    Parameters
    ----------
    n : int
        The number of qubits.

    Returns
    -------
    np.ndarray
        The W state vector.
    """
    if n < 1:
        raise ValueError("Number of qubits must be at least 1.")
    dim = 1 << n
    vec = np.zeros(dim, dtype=complex)
    norm = 1.0 / np.sqrt(n)
    for i in range(n):
        idx = 1 << i
        vec[idx] = norm
    return vec


def partial_trace(state: np.ndarray, keep_qubits: list[int], num_qubits: int) -> np.ndarray:
    """
    Compute the reduced density matrix by tracing out the qubits not in keep_qubits.

    Parameters
    ----------
    state : np.ndarray
        The state vector or density matrix.
    keep_qubits : list[int]
        The list of qubit indices to keep.
    num_qubits : int
        The total number of qubits.

    Returns
    -------
    np.ndarray
        The reduced density matrix.
    """
    if state.ndim == 1:
        rho = np.outer(state, np.conj(state))
    else:
        rho = state
    
    trace_qubits = [i for i in range(num_qubits) if i not in keep_qubits]
    if not trace_qubits:
        return rho
        
    reshaped_rho = rho.reshape([2] * (2 * num_qubits))
    
    for count, q in enumerate(trace_qubits):
        axis1 = q - count
        axis2 = q - count + num_qubits - count
        reshaped_rho = np.trace(reshaped_rho, axis1=axis1, axis2=axis2)
        
    dim = 1 << len(keep_qubits)
    return reshaped_rho.reshape(dim, dim)


def schmidt_decompose(state: np.ndarray, partition_a: list[int], num_qubits: int) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Perform the Schmidt decomposition of a pure state for a given bipartition.

    Parameters
    ----------
    state : np.ndarray
        The state vector.
    partition_a : list[int]
        The qubit indices for partition A.
    num_qubits : int
        The total number of qubits.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
        Schmidt coefficients, basis vectors for A, and basis vectors for B.
    """
    partition_b = [i for i in range(num_qubits) if i not in partition_a]
    perm = partition_a + partition_b
    
    tensor = state.reshape([2] * num_qubits)
    tensor = np.transpose(tensor, perm)
    
    dim_a = 1 << len(partition_a)
    dim_b = 1 << len(partition_b)
    
    matrix = tensor.reshape(dim_a, dim_b)
    
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    basis_a = [U[:, i] for i in range(len(S))]
    basis_b = [Vh[i, :] for i in range(len(S))]
    
    return S, basis_a, basis_b


def entanglement_entropy(state: np.ndarray, partition_a: list[int], num_qubits: int) -> float:
    """
    Compute the von Neumann entanglement entropy for a given bipartition.

    Parameters
    ----------
    state : np.ndarray
        The state vector or density matrix.
    partition_a : list[int]
        The qubit indices for partition A.
    num_qubits : int
        The total number of qubits.

    Returns
    -------
    float
        The von Neumann entanglement entropy.
    """
    if state.ndim == 1:
        S, _, _ = schmidt_decompose(state, partition_a, num_qubits)
        eigenvalues = S ** 2
    else:
        rho_a = partial_trace(state, partition_a, num_qubits)
        eigenvalues = np.linalg.eigvalsh(rho_a)
        
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    return -float(np.sum(eigenvalues * np.log2(eigenvalues)))


def overlap(psi: np.ndarray, phi: np.ndarray) -> complex:
    """
    Compute the inner product (overlap) <psi|phi> of two states.

    Parameters
    ----------
    psi : np.ndarray
        The bra vector.
    phi : np.ndarray
        The ket vector.

    Returns
    -------
    complex
        The inner product.
    """
    return complex(np.vdot(psi, phi))


def normalize(state: np.ndarray) -> np.ndarray:
    """
    Normalize the given state vector.

    Parameters
    ----------
    state : np.ndarray
        The state vector to normalize.

    Returns
    -------
    np.ndarray
        The normalized state vector.
    """
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return state / norm


def plus_state(num_qubits: int) -> np.ndarray:
    """
    Return the n-qubit |+...+> state.

    Parameters
    ----------
    num_qubits : int
        The number of qubits.

    Returns
    -------
    np.ndarray
        The |+...+> state vector.
    """
    dim = 1 << num_qubits
    return np.ones(dim, dtype=complex) / np.sqrt(dim)


def minus_state(num_qubits: int) -> np.ndarray:
    """
    Return the n-qubit |-...-> state.

    Parameters
    ----------
    num_qubits : int
        The number of qubits.

    Returns
    -------
    np.ndarray
        The |-...-> state vector.
    """
    minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
    if num_qubits == 0:
        return np.array([1.0], dtype=complex)
    
    return kron(*(minus for _ in range(num_qubits)))


def product_state(*single_qubit_states: np.ndarray) -> np.ndarray:
    """
    Return the tensor product of the given single-qubit states.

    Parameters
    ----------
    single_qubit_states : np.ndarray
        The individual single-qubit state vectors.

    Returns
    -------
    np.ndarray
        The resulting product state vector.
    """
    return kron(*single_qubit_states)


def state_to_density_matrix(state: np.ndarray) -> np.ndarray:
    """
    Convert a state vector to a density matrix |psi><psi|.

    Parameters
    ----------
    state : np.ndarray
        The state vector.

    Returns
    -------
    np.ndarray
        The density matrix.
    """
    if state.ndim == 2:
        return state
    return np.outer(state, np.conj(state))


def purity(state: np.ndarray) -> float:
    """
    Compute the purity Tr(rho^2) of a state.

    Parameters
    ----------
    state : np.ndarray
        The state vector or density matrix.

    Returns
    -------
    float
        The purity.
    """
    if state.ndim == 1:
        return 1.0
    rho_sq = np.matmul(state, state)
    return float(np.real(np.trace(rho_sq)))
