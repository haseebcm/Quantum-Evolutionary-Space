"""Simple quantum circuit builder and executor (algebraic model).

Circuit composes gates and applies them to an N-qubit state vector using the
matrix representations from quantum_gates. This is a pure Python linear
algebra model of quantum circuits suitable for testing, reasoning, and
unit tests — not a hardware backend.
"""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field

import numpy as np

from .multi_qubit import zero_state
from .quantum_gates import (
    SX,
    H,
    Phase,
    Rx,
    Ry,
    Rz,
    S,
    T,
    X,
    Y,
    Z,
    cnot_matrix,
    controlled,
    cy_matrix,
    cz_matrix,
    fredkin_matrix,
    swap_matrix,
    toffoli_matrix,
)


@dataclass
class GateOp:
    name: str
    matrix: np.ndarray | None
    targets: tuple[int, ...] | None = None


@dataclass
class Circuit:
    num_qubits: int
    ops: list[GateOp] = field(default_factory=list)
    max_state_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.num_qubits, int) or isinstance(self.num_qubits, bool) or self.num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer")
        if not isinstance(self.max_state_bytes, int) or self.max_state_bytes <= 0:
            raise ValueError("max_state_bytes must be a positive integer")
        # Check bit length before shifting to bound even maliciously large counts.
        if self.num_qubits > self.max_state_bytes.bit_length() or 16 * (1 << self.num_qubits) > self.max_state_bytes:
            raise ValueError("state vector exceeds configured memory limit")

    def _apply_local(self, state: np.ndarray, op: GateOp) -> np.ndarray:
        if op.targets is None:
            return op.matrix @ state
        targets = op.targets
        self._check_qubits(*targets)
        order = list(targets) + [q for q in range(self.num_qubits) if q not in targets]
        tensor = state.reshape([2] * self.num_qubits).transpose(order)
        evolved = op.matrix @ tensor.reshape(1 << len(targets), -1)
        return evolved.reshape([2] * self.num_qubits).transpose(np.argsort(order)).reshape(-1)

    def _check_qubit(self, q: int) -> None:
        if not (0 <= q < self.num_qubits):
            raise IndexError("qubit index out of range")

    def _check_qubits(self, *qs: int) -> None:
        for q in qs:
            self._check_qubit(q)
        if len(set(qs)) != len(qs):
            raise ValueError("qubits must be distinct")

    def apply_x(self, target: int) -> None:
        """Apply an X (NOT) gate to the target qubit."""
        self._check_qubit(target)
        mat = X
        self.ops.append(GateOp(f"X_{target}", mat, (target,)))

    def apply_y(self, target: int) -> None:
        """Apply a Y gate to the target qubit."""
        self._check_qubit(target)
        mat = Y
        self.ops.append(GateOp(f"Y_{target}", mat, (target,)))

    def apply_z(self, target: int) -> None:
        """Apply a Z (Phase-flip) gate to the target qubit."""
        self._check_qubit(target)
        mat = Z
        self.ops.append(GateOp(f"Z_{target}", mat, (target,)))

    def apply_h(self, target: int) -> None:
        """Apply a Hadamard gate to the target qubit."""
        self._check_qubit(target)
        mat = H
        self.ops.append(GateOp(f"H_{target}", mat, (target,)))

    def apply_s(self, target: int) -> None:
        """Apply an S (Phase) gate to the target qubit."""
        self._check_qubit(target)
        mat = S
        self.ops.append(GateOp(f"S_{target}", mat, (target,)))

    def apply_t(self, target: int) -> None:
        """Apply a T (pi/8) gate to the target qubit."""
        self._check_qubit(target)
        mat = T
        self.ops.append(GateOp(f"T_{target}", mat, (target,)))

    def apply_sx(self, target: int) -> None:
        """Apply an SX (sqrt-X) gate to the target qubit."""
        self._check_qubit(target)
        mat = SX
        self.ops.append(GateOp(f"SX_{target}", mat, (target,)))

    def apply_rx(self, target: int, theta: float) -> None:
        """Apply an Rx rotation by theta to the target qubit."""
        self._check_qubit(target)
        mat = Rx(theta)
        self.ops.append(GateOp(f"Rx({theta:.2f})_{target}", mat, (target,)))

    def apply_ry(self, target: int, theta: float) -> None:
        """Apply an Ry rotation by theta to the target qubit."""
        self._check_qubit(target)
        mat = Ry(theta)
        self.ops.append(GateOp(f"Ry({theta:.2f})_{target}", mat, (target,)))

    def apply_rz(self, target: int, theta: float) -> None:
        """Apply an Rz rotation by theta to the target qubit."""
        self._check_qubit(target)
        mat = Rz(theta)
        self.ops.append(GateOp(f"Rz({theta:.2f})_{target}", mat, (target,)))

    def apply_phase(self, target: int, theta: float) -> None:
        """Apply a Phase rotation by theta to the target qubit."""
        self._check_qubit(target)
        mat = Phase(theta)
        self.ops.append(GateOp(f"Phase({theta:.2f})_{target}", mat, (target,)))

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply a Controlled-NOT gate."""
        self._check_qubits(control, target)
        two = cnot_matrix()
        mat = two
        self.ops.append(GateOp(f"CNOT_{control}_{target}", mat, (control, target)))

    def apply_cz(self, control: int, target: int) -> None:
        """Apply a Controlled-Z gate."""
        self._check_qubits(control, target)
        two = cz_matrix()
        mat = two
        self.ops.append(GateOp(f"CZ_{control}_{target}", mat, (control, target)))

    def apply_cy(self, control: int, target: int) -> None:
        """Apply a Controlled-Y gate."""
        self._check_qubits(control, target)
        two = cy_matrix()
        mat = two
        self.ops.append(GateOp(f"CY_{control}_{target}", mat, (control, target)))

    def apply_swap(self, q1: int, q2: int) -> None:
        """Apply a SWAP gate."""
        self._check_qubits(q1, q2)
        two = swap_matrix()
        mat = two
        self.ops.append(GateOp(f"SWAP_{q1}_{q2}", mat, (q1, q2)))

    def apply_controlled_u(self, control: int, target: int, gate: np.ndarray) -> None:
        """Apply an arbitrary Controlled-U gate."""
        self._check_qubits(control, target)
        gate = np.asarray(gate, dtype=complex)
        if gate.shape != (2, 2) or not np.all(np.isfinite(gate)) or not np.allclose(gate.conj().T @ gate, np.eye(2)):
            raise ValueError("controlled gate must be a finite 2x2 unitary")
        c_gate = controlled(gate)
        mat = c_gate
        self.ops.append(GateOp(f"CU_{control}_{target}", mat, (control, target)))

    def apply_toffoli(self, c1: int, c2: int, target: int) -> None:
        """Apply a Toffoli (CCX) gate."""
        self._check_qubits(c1, c2, target)
        three = toffoli_matrix()
        mat = three
        self.ops.append(GateOp(f"CCX_{c1}_{c2}_{target}", mat, (c1, c2, target)))

    def apply_fredkin(self, control: int, t1: int, t2: int) -> None:
        """Apply a Fredkin (CSWAP) gate."""
        self._check_qubits(control, t1, t2)
        three = fredkin_matrix()
        mat = three
        self.ops.append(GateOp(f"CSWAP_{control}_{t1}_{t2}", mat, (control, t1, t2)))

    def barrier(self) -> None:
        """Adds a visual barrier (no-op)."""
        self.ops.append(GateOp("BARRIER", None))

    def run(self, state: np.ndarray | None = None) -> np.ndarray:
        """Execute the circuit on an initial state vector."""
        if state is None:
            state = zero_state(self.num_qubits)
        s = np.asarray(state, dtype=complex)
        expected_dim = 1 << self.num_qubits
        if s.shape != (expected_dim,) or not np.all(np.isfinite(s)):
            raise ValueError(
                f"state vector has size {s.size} but expected {expected_dim} for {self.num_qubits} qubits"
            )
        for op in self.ops:
            if op.matrix is not None:
                s = self._apply_local(s, op)
        return s

    def measure_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Return the probability distribution of outcomes for the given state."""
        s = np.asarray(state, dtype=complex)
        if s.shape != (1 << self.num_qubits,) or not np.all(np.isfinite(s)):
            raise ValueError("state must be a finite vector matching the circuit")
        probs = np.abs(s) ** 2
        total = probs.sum()
        if total == 0:
            raise ValueError("state has zero norm")
        probs /= total
        return probs

    def measure(
        self, num_shots: int = 1024, rng: random.Random | None = None, state: np.ndarray | None = None
    ) -> dict[str, int]:
        """Simulate measuring the circuit multiple times.
        
        Returns a dictionary mapping bitstrings to their outcome frequencies.
        """
        final_state = self.run(state)
        probs = self.measure_probabilities(final_state)

        if rng is None:
            rng = random.Random()

        dim = 1 << self.num_qubits
        choices = list(range(dim))
        outcomes = rng.choices(choices, weights=list(probs), k=num_shots)

        hist: dict[str, int] = collections.defaultdict(int)
        fmt = f"0{self.num_qubits}b"
        for out in outcomes:
            hist[format(out, fmt)] += 1
        return dict(hist)

    def measure_qubit(self, target: int, state: np.ndarray, rng: random.Random | None = None) -> tuple[int, np.ndarray]:
        """Measure a single qubit, returning the outcome and the collapsed state."""
        self._check_qubit(target)
        if rng is None:
            rng = random.Random()

        s = np.asarray(state, dtype=complex)
        probs = self.measure_probabilities(s)

        p0 = 0.0
        dim = 1 << self.num_qubits
        for i in range(dim):
            if (i >> (self.num_qubits - 1 - target)) & 1 == 0:
                p0 += probs[i]

        outcome = 0 if rng.random() < p0 else 1

        collapsed = np.zeros_like(s)
        for i in range(dim):
            bit = (i >> (self.num_qubits - 1 - target)) & 1
            if bit == outcome:
                collapsed[i] = s[i]

        norm = np.linalg.norm(collapsed)
        if norm > 0:
            collapsed /= norm

        return outcome, collapsed

    def depth(self) -> int:
        """Calculate the circuit depth (critical path length)."""
        qubit_depth = [0] * self.num_qubits
        for op in self.ops:
            if op.matrix is None:
                continue
            parts = op.name.split("_")
            involved = []
            for p in parts[1:]:
                try:
                    involved.append(int(p))
                except ValueError:
                    pass
            if not involved:
                involved = list(range(self.num_qubits))

            max_d = max(qubit_depth[q] for q in involved) if involved else 0
            for q in involved:
                qubit_depth[q] = max_d + 1
        return max(qubit_depth) if qubit_depth else 0

    def gate_count(self) -> int:
        """Return the number of gates (excluding barriers)."""
        return sum(1 for op in self.ops if op.matrix is not None)

    def gate_histogram(self) -> dict[str, int]:
        """Return a histogram of gate types used."""
        hist: dict[str, int] = collections.defaultdict(int)
        for op in self.ops:
            if op.matrix is not None:
                gate_type = op.name.split("_")[0]
                gate_type = gate_type.split("(")[0]
                hist[gate_type] += 1
        return dict(hist)

    def compose(self, other: Circuit) -> Circuit:
        """Compose this circuit with another by appending its operations."""
        if self.num_qubits != other.num_qubits:
            raise ValueError("Circuits must have the same number of qubits to be composed")
        c = Circuit(self.num_qubits, max_state_bytes=self.max_state_bytes)
        c.ops = self.ops + other.ops
        return c

    def repeat(self, n: int) -> Circuit:
        """Return a new circuit repeating the operations n times."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("repeat count must be a nonnegative integer")
        c = Circuit(self.num_qubits, max_state_bytes=self.max_state_bytes)
        c.ops = self.ops * n
        return c

    def inverse(self) -> Circuit:
        """Return a new circuit applying the inverse operations in reverse order."""
        c = Circuit(self.num_qubits, max_state_bytes=self.max_state_bytes)
        for op in reversed(self.ops):
            if op.matrix is None:
                c.ops.append(op)
            else:
                c.ops.append(GateOp(op.name + "_dag", op.matrix.conj().T, op.targets))
        return c

    def draw(self) -> str:
        """Return an ASCII art representation showing qubits as horizontal lines with gate labels."""
        lines = [f"q{i}: " for i in range(self.num_qubits)]
        for op in self.ops:
            if op.matrix is None:
                for i in range(self.num_qubits):
                    lines[i] += "-||-"
                continue

            parts = op.name.split("_")
            base_name = parts[0]
            involved = []
            for p in parts[1:]:
                try:
                    involved.append(int(p))
                except ValueError:
                    pass

            if not involved:
                involved = list(range(self.num_qubits))

            gate_len = len(base_name) + 4
            for i in range(self.num_qubits):
                if i in involved:
                    if len(involved) == 1:
                        chunk = f"-[{base_name}]-"
                    elif i == involved[0]:
                        chunk = "-*" + "-" * (gate_len - 2)
                    elif len(involved) > 2 and i == involved[1]:
                        chunk = "-*" + "-" * (gate_len - 2)
                    else:
                        chunk = f"-({base_name})-"

                    if len(chunk) < gate_len:
                        chunk = chunk.ljust(gate_len, "-")
                    elif len(chunk) > gate_len:
                        chunk = chunk[:gate_len]

                    lines[i] += chunk
                else:
                    lines[i] += "-" * gate_len

        return "\n".join(lines)

    def get_amplitude(self, state: np.ndarray, bitstring: str) -> complex:
        """Get the probability amplitude of a given bitstring from a state vector."""
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring must be length {self.num_qubits}")
        idx = int(bitstring, 2)
        return state[idx]

    def concurrence(self, state: np.ndarray) -> float:
        """Calculate the concurrence (entanglement measure) of a 2-qubit state."""
        if self.num_qubits != 2:
            raise ValueError("Concurrence is only defined for 2 qubits")
        s = np.asarray(state, dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]])
        sy_sy = np.kron(sy, sy)

        s_conj = s.conj()
        val = s_conj.T @ sy_sy @ s_conj
        return float(np.abs(val))

    def reset(self) -> None:
        """Reset the circuit by clearing all operations."""
        self.ops.clear()
