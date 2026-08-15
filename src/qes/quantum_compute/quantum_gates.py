"""Common quantum gates as unitary matrices and helpers to apply them.

These functions provide matrix representations for single- and two-qubit
standard gates (X, H, Z, S, T, CNOT) and helpers to expand them to act on
an N-qubit state vector by tensoring with identities. These are linear
algebra constructs and do not control real quantum hardware.
"""
from __future__ import annotations

import numpy as np

# Single-qubit gates
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def Rx(theta: float) -> np.ndarray:
    """Return the Rx rotation gate matrix for angle `theta`."""
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    """Return the Ry rotation gate matrix for angle `theta`."""
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    """Return the Rz rotation gate matrix for angle `theta`."""
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


def Phase(theta: float) -> np.ndarray:
    """Return the Phase gate matrix for angle `theta`."""
    return np.array([[1, 0],
                     [0, np.exp(1j * theta)]], dtype=complex)


def cnot_matrix() -> np.ndarray:
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def cz_matrix() -> np.ndarray:
    """Return the 4x4 controlled-Z gate matrix."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)


def cy_matrix() -> np.ndarray:
    """Return the 4x4 controlled-Y gate matrix."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, -1j],
                     [0, 0, 1j, 0]], dtype=complex)


def swap_matrix() -> np.ndarray:
    """Return the 4x4 SWAP gate matrix."""
    return np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)


def toffoli_matrix() -> np.ndarray:
    """Return the 8x8 Toffoli (CCX) gate matrix."""
    m = np.eye(8, dtype=complex)
    m[6, 6] = 0
    m[7, 7] = 0
    m[6, 7] = 1
    m[7, 6] = 1
    return m


def fredkin_matrix() -> np.ndarray:
    """Return the 8x8 Fredkin (CSWAP) gate matrix."""
    m = np.eye(8, dtype=complex)
    m[5, 5] = 0
    m[6, 6] = 0
    m[5, 6] = 1
    m[6, 5] = 1
    return m


def controlled(gate: np.ndarray) -> np.ndarray:
    """Return the 4x4 controlled-U gate matrix for a 2x2 single-qubit `gate`."""
    if gate.shape != (2, 2):
        raise ValueError("controlled() expects a 2x2 unitary matrix")
    result = np.eye(4, dtype=complex)
    result[2:4, 2:4] = gate
    return result


def compose(*gates: np.ndarray) -> np.ndarray:
    """Compose multiple gate matrices by multiplying them in sequence.
    
    The order is such that compose(A, B, C) returns C @ B @ A, matching the
    circuit application order where A is applied first.
    """
    if not gates:
        raise ValueError("At least one gate must be provided to compose()")
    result = gates[0]
    for gate in gates[1:]:
        result = gate @ result
    return result


def is_unitary(gate: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if a given matrix is unitary within the specified tolerance."""
    gate = np.asarray(gate, dtype=complex)
    identity = np.eye(gate.shape[0], dtype=complex)
    return bool(np.allclose(gate.conj().T @ gate, identity, atol=atol))


def gate_fidelity(U: np.ndarray, V: np.ndarray) -> float:
    """Calculate the gate fidelity between two unitary matrices U and V.
    
    Calculated as |Tr(U^dagger V)|^2 / (dim^2).
    """
    dim = U.shape[0]
    if V.shape[0] != dim:
        raise ValueError("Matrices must have the same dimensions")
    trace = np.trace(U.conj().T @ V)
    return float(np.abs(trace)**2 / (dim**2))


def gate_power(gate: np.ndarray, n: float) -> np.ndarray:
    """Return the gate raised to the power `n` via eigendecomposition."""
    gate = np.asarray(gate, dtype=complex)
    evals, evecs = np.linalg.eig(gate)
    return evecs @ np.diag(evals ** n) @ np.linalg.inv(evecs)


def parametric_gate(name: str, theta: float) -> np.ndarray:
    """Return a parametric rotation gate by name ('rx', 'ry', 'rz', 'phase')."""
    name = name.lower()
    if name == "rx":
        return Rx(theta)
    elif name == "ry":
        return Ry(theta)
    elif name == "rz":
        return Rz(theta)
    elif name == "phase":
        return Phase(theta)
    else:
        raise ValueError(f"Unknown parametric gate name: {name}")


def expand_single_qubit_gate(gate: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
    """Return the full 2^n x 2^n matrix for `gate` applied to `target` qubit.

    Validates inputs and treats qubit 0 as the most-significant tensor factor.
    """
    if not (0 <= target < num_qubits):
        raise IndexError("target qubit index out of range")
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    mats: list[np.ndarray] = []
    for i in range(num_qubits):
        mats.append(gate if i == target else IDENTITY)
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _permutation_matrix(perm: list[int]) -> np.ndarray:
    """Return the permutation matrix that reorders tensor axes according to `perm`.

    `perm` must be a permutation of range(n). If applied as `a.transpose(perm)`,
    the resulting flattened state vector equals P @ flattened_original.
    """
    n = len(perm)
    dim = 1 << n
    P = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        # multi-index for basis vector i (most-significant bit first)
        m = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        m_permuted = [m[perm[k]] for k in range(n)]
        j = 0
        for bit in m_permuted:
            j = (j << 1) | bit
        P[j, i] = 1.0
    return P


def _expand_adjacent_two_qubit_gate(gate: np.ndarray, lo: int, num_qubits: int) -> np.ndarray:
    """Expand a 4x4 gate placed at adjacent positions (lo, lo+1).

    Uses direct tensoring of gate at position `lo` with identities elsewhere.
    """
    mats: list[np.ndarray] = []
    i = 0
    while i < num_qubits:
        if i == lo:
            mats.append(gate)
            i += 2
            continue
        mats.append(IDENTITY)
        i += 1
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def expand_two_qubit_gate(gate: np.ndarray, control: int, target: int, num_qubits: int) -> np.ndarray:
    """Expand a 4x4 two-qubit gate matrix (control,target) into the full space.

    Works for adjacent and non-adjacent qubits by permuting axes, applying the
    two-qubit gate on adjacent factors, and permuting back. Qubit 0 is the
    most-significant tensor factor.
    """
    if control == target:
        raise ValueError("control and target must differ")
    if not (0 <= control < num_qubits) or not (0 <= target < num_qubits):
        raise IndexError("control/target index out of range")
    if num_qubits < 2:
        raise ValueError("num_qubits must be >= 2")

    lo, hi = min(control, target), max(control, target)
    if hi == lo + 1:
        # adjacent — simple expansion
        return _expand_adjacent_two_qubit_gate(gate, lo, num_qubits)

    # Non-adjacent: build a permutation that moves (control,target) to (lo, lo+1)
    axes = [i for i in range(num_qubits) if i not in (control, target)]
    # Insert control then target at position `lo` to preserve the requested order
    new_axes = axes[:lo] + [control, target] + axes[lo:]
    P = _permutation_matrix(new_axes)
    # Build adjacent operator in permuted basis
    U_adj = _expand_adjacent_two_qubit_gate(gate, lo, num_qubits)
    # Conjugate: U = P^T * U_adj * P
    return P.conj().T @ U_adj @ P


def _expand_adjacent_three_qubit_gate(gate: np.ndarray, lo: int, num_qubits: int) -> np.ndarray:
    """Expand an 8x8 gate placed at adjacent positions (lo, lo+1, lo+2)."""
    mats: list[np.ndarray] = []
    i = 0
    while i < num_qubits:
        if i == lo:
            mats.append(gate)
            i += 3
            continue
        mats.append(IDENTITY)
        i += 1
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def expand_three_qubit_gate(gate: np.ndarray, q0: int, q1: int, q2: int, num_qubits: int) -> np.ndarray:
    """Expand an 8x8 three-qubit gate matrix (q0, q1, q2) into the full space.
    
    Qubit 0 is the most-significant tensor factor.
    """
    if len(set([q0, q1, q2])) != 3:
        raise ValueError("Qubit indices must be distinct")
    if not all(0 <= q < num_qubits for q in (q0, q1, q2)):
        raise IndexError("Qubit index out of range")
    if num_qubits < 3:
        raise ValueError("num_qubits must be >= 3")

    targets = (q0, q1, q2)
    lo = min(targets)
    
    # Check if adjacent and ordered
    if (q0, q1, q2) == (lo, lo + 1, lo + 2):
        return _expand_adjacent_three_qubit_gate(gate, lo, num_qubits)

    # Non-adjacent or out of order: build permutation
    axes = [i for i in range(num_qubits) if i not in targets]
    new_axes = axes[:lo] + [q0, q1, q2] + axes[lo:]
    P = _permutation_matrix(new_axes)
    U_adj = _expand_adjacent_three_qubit_gate(gate, lo, num_qubits)
    return P.conj().T @ U_adj @ P


def apply_gate(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Apply a full-space unitary matrix to a state vector."""
    state = np.asarray(state, dtype=complex)
    return gate.dot(state)
