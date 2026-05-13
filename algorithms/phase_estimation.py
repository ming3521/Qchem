"""Quantum phase estimation utilities.

This module provides small circuit builders and classical reference routines for
quantum phase estimation (QPE) in the HYQ ``QuantumCircuit`` format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

from backends.core import QuantumCircuit


@dataclass
class PhaseEstimationResult:
    phases: np.ndarray
    probabilities: np.ndarray
    energies: Optional[np.ndarray] = None
    bitstrings: Optional[List[str]] = None


def inverse_qft(qc: QuantumCircuit, qubits: Sequence[int], do_swaps: bool = True):
    """Append an inverse QFT on ``qubits``.

    Qubit order follows the project convention that lower-index qubits appear
    earlier in measured bitstrings.  ``qft`` and ``inverse_qft`` are exact
    inverses when called with the same qubit order and ``do_swaps`` value.
    """

    qubits = list(qubits)
    n = len(qubits)
    if do_swaps:
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
    for j in range(n):
        q = qubits[j]
        for m in range(j):
            angle = -np.pi / (2 ** (j - m))
            qc.cp(qubits[m], q, angle)
        qc.h(q)
    return qc


def qft(qc: QuantumCircuit, qubits: Sequence[int], do_swaps: bool = True):
    """Append a QFT on ``qubits``."""

    qubits = list(qubits)
    n = len(qubits)
    for j in reversed(range(n)):
        q = qubits[j]
        qc.h(q)
        for m in reversed(range(j)):
            angle = np.pi / (2 ** (j - m))
            qc.cp(qubits[m], q, angle)
    if do_swaps:
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
    return qc


def _copy_instruction_with_controls(
    qc: QuantumCircuit,
    inst,
    qubit_map: Mapping[int, int],
    extra_controls: Sequence[int] = (),
    extra_control_values: Optional[Sequence[int]] = None,
):
    """Copy one HYQ instruction into ``qc`` under a qubit map.

    ``extra_controls`` is used by QPE to add the ancilla control.  Existing
    controls on the copied instruction are preserved.
    """

    if extra_control_values is None:
        extra_control_values = [1] * len(extra_controls)
    extra_controls = list(extra_controls)
    extra_control_values = list(extra_control_values)

    # Inline sub-circuits recursively.  This avoids relying on backend support
    # for controlled arbitrary sub-circuit instructions.
    if getattr(inst, "circuit", None) is not None:
        sub = inst.circuit
        mapped_targets = [qubit_map[int(q)] for q in inst.qubits]
        sub_map = {local_q: mapped_targets[local_q] for local_q in range(sub.n_qubits)}
        original_controls = [qubit_map[int(q)] for q in getattr(inst, "control_qubits", [])]
        original_values = list(getattr(inst, "control_values", []) or [1] * len(original_controls))
        controls = original_controls + extra_controls
        values = original_values + extra_control_values
        for sub_inst in sub.instructions:
            _copy_instruction_with_controls(qc, sub_inst, sub_map, controls, values)
        return None

    targets = [qubit_map[int(q)] for q in getattr(inst, "qubits", [])]

    if getattr(inst, "matrix", None) is not None:
        if not hasattr(qc, "unitary"):
            raise NotImplementedError("The target QuantumCircuit does not support unitary(matrix, qubits).")
        copied = qc.unitary(np.asarray(inst.matrix, dtype=np.complex128), targets, check_unitary=False)
    else:
        name = getattr(inst, "name", None)
        params = list(getattr(inst, "params", []) or [])
        if name == "GlobalPhase":
            if extra_controls:
                raise NotImplementedError("Controlled global phase is not supported by this QPE builder.")
            copied = qc.global_phase(params[0] if params else 0.0)
        else:
            copied = qc.append(name, targets, params=params)

    original_controls = [qubit_map[int(q)] for q in getattr(inst, "control_qubits", [])]
    original_values = list(getattr(inst, "control_values", []) or [1] * len(original_controls))
    if original_controls:
        copied.control(original_controls, original_values)
    if extra_controls:
        copied.control(extra_controls, extra_control_values)
    return copied


def build_qpe_circuit(
    unitary_circuit: QuantumCircuit,
    n_ancilla: int,
    target_qubits: Optional[Sequence[int]] = None,
) -> QuantumCircuit:
    """Build standard QPE for a unitary represented as a HYQ sub-circuit.

    The returned circuit contains ``n_ancilla`` counting qubits followed by the
    target register by default.  The caller is responsible for preparing the
    target register in an eigenstate of ``unitary_circuit`` before running QPE.

    Measured ancilla bitstrings can be passed directly to ``phases_from_counts``.
    """

    if n_ancilla <= 0:
        raise ValueError("n_ancilla must be positive")

    if target_qubits is None:
        target_qubits = list(range(n_ancilla, n_ancilla + unitary_circuit.n_qubits))
    target_qubits = [int(q) for q in target_qubits]
    if len(target_qubits) != unitary_circuit.n_qubits:
        raise ValueError("target_qubits length must equal unitary_circuit.n_qubits")
    if len(set(target_qubits)) != len(target_qubits):
        raise ValueError("target_qubits must not contain duplicates")
    if any(q < 0 for q in target_qubits):
        raise ValueError("target_qubits must be non-negative")

    anc = list(range(n_ancilla))
    if set(anc).intersection(target_qubits):
        raise ValueError("target_qubits must not overlap the ancilla register 0..n_ancilla-1")

    n_total = max(n_ancilla + unitary_circuit.n_qubits, max(target_qubits) + 1)
    qc = QuantumCircuit(n_total, name="QPE")
    qc.h(anc)

    qubit_map = {q: target_qubits[q] for q in range(unitary_circuit.n_qubits)}
    for k, a in enumerate(anc):
        # With ancilla bitstrings read as q0 q1 ... q_{t-1}, qubit q0 is the
        # most significant phase bit.  Applying U^(2^k) and then inverse QFT on
        # reversed ancillas yields bitstrings compatible with phases_from_counts.
        repeats = 2**k
        for _ in range(repeats):
            for inst in unitary_circuit.instructions:
                _copy_instruction_with_controls(qc, inst, qubit_map, extra_controls=[a])

    # The reversed order is important: without it, the output bitstrings are
    # bit-reversed relative to int(bitstring, 2) / 2**n_ancilla.
    inverse_qft(qc, list(reversed(anc)), do_swaps=False)
    return qc


def phases_from_counts(counts, n_ancilla: int, *, reverse_bits: bool = False) -> PhaseEstimationResult:
    """Convert measured QPE counts into phases and probabilities.

    Set ``reverse_bits=True`` only when processing counts produced by older QPE
    circuits whose ancilla register was bit-reversed.
    """

    if n_ancilla <= 0:
        raise ValueError("n_ancilla must be positive")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one positive count")

    bitstrings = sorted(counts, key=counts.get, reverse=True)
    for b in bitstrings:
        if len(b) != n_ancilla or any(ch not in "01" for ch in b):
            raise ValueError(f"Invalid bitstring {b!r} for n_ancilla={n_ancilla}")

    probs = np.array([counts[b] / total for b in bitstrings], dtype=float)
    read_bits = [b[::-1] if reverse_bits else b for b in bitstrings]
    phases = np.array([int(b, 2) / (2**n_ancilla) for b in read_bits], dtype=float)
    return PhaseEstimationResult(phases=phases, probabilities=probs, bitstrings=bitstrings)


def _qpe_kernel_probability(phase: float, grid: np.ndarray) -> np.ndarray:
    """Return exact finite-register QPE probabilities for one eigenphase."""

    m = grid.size
    delta = phase - grid
    # Use the closed form of the geometric sum.  The near-integer branch avoids
    # numerical 0/0 when the phase is exactly representable on the grid.
    numer = np.sin(np.pi * m * delta)
    denom = np.sin(np.pi * delta)
    probs = np.empty_like(grid, dtype=float)
    close = np.isclose(denom, 0.0, atol=1e-14)
    probs[close] = 1.0
    probs[~close] = (numer[~close] / (m * denom[~close])) ** 2
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return probs


def exact_phase_estimation_from_unitary(
    unitary: np.ndarray,
    input_state: np.ndarray,
    n_ancilla: int,
    energy_scale: Optional[float] = None,
) -> PhaseEstimationResult:
    """Classical reference for finite-register QPE output probabilities.

    Unlike a nearest-grid approximation, this routine returns the true QPE
    probability distribution for ``n_ancilla`` counting qubits.  If an eigenphase
    is not exactly representable with ``n_ancilla`` bits, probability is spread
    over multiple bitstrings according to the Dirichlet kernel.
    """

    if n_ancilla <= 0:
        raise ValueError("n_ancilla must be positive")

    unitary = np.asarray(unitary, dtype=np.complex128)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("unitary must be a square matrix")

    input_state = np.asarray(input_state, dtype=np.complex128).reshape(-1)
    if input_state.shape[0] != unitary.shape[0]:
        raise ValueError("input_state dimension must match unitary")
    norm = np.linalg.norm(input_state)
    if norm <= 0:
        raise ValueError("input_state must have non-zero norm")
    input_state = input_state / norm

    vals, vecs = np.linalg.eig(unitary)
    phases = (np.angle(vals) / (2 * np.pi)) % 1.0

    # For unitary matrices, eigenvectors form an orthonormal basis up to numerical
    # error.  Keep the eigenvectors paired with their eigenphases; replacing them
    # by a QR basis would generally mix non-degenerate eigenspaces.
    overlaps = np.abs(vecs.conj().T @ input_state) ** 2
    if overlaps.sum() > 0:
        overlaps = overlaps / overlaps.sum()

    grid = np.arange(2**n_ancilla, dtype=float) / (2**n_ancilla)
    probs = np.zeros_like(grid, dtype=float)
    for phase, weight in zip(phases, overlaps):
        probs += float(weight.real) * _qpe_kernel_probability(float(phase), grid)
    if probs.sum() > 0:
        probs = probs / probs.sum()

    energies = None
    if energy_scale is not None:
        if energy_scale == 0:
            raise ValueError("energy_scale must be non-zero")
        energies = 2 * np.pi * grid / energy_scale

    return PhaseEstimationResult(
        phases=grid,
        probabilities=probs,
        energies=energies,
        bitstrings=[format(i, f"0{n_ancilla}b") for i in range(2**n_ancilla)],
    )


def iterative_phase_estimation_step(expectation_cos: float, expectation_sin: float) -> float:
    """Return a phase estimate from Hadamard-test cosine/sine estimates."""

    return float(np.arctan2(expectation_sin, expectation_cos) / (2 * np.pi) % 1.0)
