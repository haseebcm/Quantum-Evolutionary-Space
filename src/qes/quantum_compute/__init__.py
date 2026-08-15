"""Quantum-inspired compute primitives for QES.

This package provides computational, simulated constructs inspired by the
architecture document: ComputationalQubit, per-layer managers, quantum circuit
simulation, ACROS correction, evolutionary dynamics, QEL streams, QSEE-11L
coordination, synchronization, and serialization utilities.

These are software-only abstractions for layered possibility-space evolution
and are explicitly not physical quantum code. Every module is production-grade
with comprehensive type hints, docstrings, and backward-compatible APIs.
"""
from __future__ import annotations

__version__ = "2.0.0"

# ── Qubit & State Representations ────────────────────────────────────────────
# ── ACROS Correction ─────────────────────────────────────────────────────────
from .acros import (
    CorrectionResult,
    acros_adam,
    acros_adaptive,
    acros_correction,
    acros_momentum,
    acros_multi_step,
    cosine_decay,
    exponential_decay,
    gradient_norm,
    make_batch_gradient_estimator,
    make_forward_gradient_estimator,
    make_gradient_estimator,
    projected_acros,
    step_decay,
)

# ── Evolution & Drift ────────────────────────────────────────────────────────
from .evolution import (
    adaptive_evolver,
    composite_evolver,
    crossover_evolver,
    exponential_decay_evolver,
    gaussian_noise_evolver,
    get_evolver,
    identity_evolver,
    linear_drift_evolver,
    list_evolvers,
    make_evolver,
    mutation_evolver,
    ou_evolver,
    probabilistic_evolver,
    register_evolver,
    sinusoidal_evolver,
    weighted_mix_evolver,
)

# ── Synchronization & Gate Utilities ─────────────────────────────────────────
from .gates import (
    cross_correlation,
    enforce_isolation,
    gated_sync,
    isolation_report,
    sync_average,
    sync_bounded,
    sync_divergence,
    sync_max_norm,
    sync_median,
    sync_momentum,
    sync_top_k,
    sync_weighted,
)

# ── Layer & QSEE-11L Coordination ───────────────────────────────────────────
from .layer import Layer
from .multi_qubit import (
    basis_state,
    bell_state,
    entanglement_entropy,
    ghz_state,
    kron,
    minus_state,
    normalize,
    overlap,
    partial_trace,
    plus_state,
    product_state,
    purity,
    random_state,
    schmidt_decompose,
    state_to_density_matrix,
    w_state,
    zero_state,
)

# ── QEL Streams ──────────────────────────────────────────────────────────────
from .qel import QELStream
from .qsee11l import QSEE11L

# ── Circuit ──────────────────────────────────────────────────────────────────
from .quantum_circuit import Circuit, GateOp

# ── Quantum Gates ────────────────────────────────────────────────────────────
from .quantum_gates import (
    IDENTITY,
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
    apply_gate,
    cnot_matrix,
    compose,
    controlled,
    cy_matrix,
    cz_matrix,
    expand_single_qubit_gate,
    expand_three_qubit_gate,
    expand_two_qubit_gate,
    fredkin_matrix,
    gate_fidelity,
    gate_power,
    is_unitary,
    parametric_gate,
    swap_matrix,
    toffoli_matrix,
)
from .qubit import ComputationalQubit

# ── Serialization ────────────────────────────────────────────────────────────
from .serialize import (
    SERIALIZE_VERSION,
    apply_diff,
    compute_checksum,
    compute_diff,
    from_bytes,
    from_compressed,
    from_file,
    from_json,
    from_jsonl,
    pretty,
    to_bytes,
    to_compressed,
    to_file,
    to_json,
    to_jsonl,
    verify_checksum,
)
from .sync import SyncManager

# Provide backward-compatible alias `I` for callers that expect the common name
I = IDENTITY  # noqa: E741

__all__ = [
    # Version
    "__version__",
    # Qubit
    "ComputationalQubit",
    # Multi-qubit states
    "kron",
    "basis_state",
    "zero_state",
    "random_state",
    "bell_state",
    "ghz_state",
    "w_state",
    "plus_state",
    "minus_state",
    "product_state",
    "normalize",
    "overlap",
    "purity",
    "state_to_density_matrix",
    "partial_trace",
    "schmidt_decompose",
    "entanglement_entropy",
    # Gates
    "X",
    "Y",
    "H",
    "Z",
    "S",
    "T",
    "SX",
    "I",
    "IDENTITY",
    "Rx",
    "Ry",
    "Rz",
    "Phase",
    "cnot_matrix",
    "cz_matrix",
    "cy_matrix",
    "swap_matrix",
    "toffoli_matrix",
    "fredkin_matrix",
    "controlled",
    "compose",
    "is_unitary",
    "gate_fidelity",
    "gate_power",
    "parametric_gate",
    "expand_single_qubit_gate",
    "expand_two_qubit_gate",
    "expand_three_qubit_gate",
    "apply_gate",
    # Circuit
    "GateOp",
    "Circuit",
    # Layer & QSEE-11L
    "Layer",
    "QSEE11L",
    # QEL
    "QELStream",
    # ACROS
    "CorrectionResult",
    "acros_correction",
    "acros_multi_step",
    "acros_adaptive",
    "acros_momentum",
    "acros_adam",
    "projected_acros",
    "make_gradient_estimator",
    "make_batch_gradient_estimator",
    "make_forward_gradient_estimator",
    "gradient_norm",
    "cosine_decay",
    "exponential_decay",
    "step_decay",
    # Evolution
    "linear_drift_evolver",
    "weighted_mix_evolver",
    "identity_evolver",
    "sinusoidal_evolver",
    "exponential_decay_evolver",
    "ou_evolver",
    "crossover_evolver",
    "mutation_evolver",
    "adaptive_evolver",
    "composite_evolver",
    "probabilistic_evolver",
    "gaussian_noise_evolver",
    "register_evolver",
    "get_evolver",
    "list_evolvers",
    "make_evolver",
    # Sync utilities
    "enforce_isolation",
    "sync_average",
    "sync_weighted",
    "sync_median",
    "sync_max_norm",
    "gated_sync",
    "isolation_report",
    "cross_correlation",
    "sync_bounded",
    "sync_momentum",
    "sync_divergence",
    "sync_top_k",
    "SyncManager",
    # Serialization
    "SERIALIZE_VERSION",
    "to_json",
    "to_file",
    "from_json",
    "from_file",
    "to_bytes",
    "from_bytes",
    "to_compressed",
    "from_compressed",
    "to_jsonl",
    "from_jsonl",
    "compute_checksum",
    "verify_checksum",
    "pretty",
    "compute_diff",
    "apply_diff",
]
