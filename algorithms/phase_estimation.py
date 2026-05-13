"""Quantum phase estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from backends.core import QuantumCircuit


@dataclass
class PhaseEstimationResult:
    phases: np.ndarray
    probabilities: np.ndarray
    energies: Optional[np.ndarray] = None
    bitstrings: Optional[List[str]] = None


def inverse_qft(qc: QuantumCircuit, qubits: Sequence[int], do_swaps: bool = True):
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


def qft(qc: QuantumCircuit, qubits: Sequence[int], do_swaps: bool = True):
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


def build_qpe_circuit(unitary_circuit: QuantumCircuit, n_ancilla: int, target_qubits: Optional[Sequence[int]] = None) -> QuantumCircuit:
    """Build standard QPE for a unitary represented as a HYQ sub-circuit.

    The current ``QuantumCircuit`` supports circuit extension but not arbitrary
    controlled sub-circuit synthesis in every backend.  This builder therefore
    appends controlled copies of each instruction in the supplied unitary circuit.
    """

    if target_qubits is None:
        target_qubits = list(range(n_ancilla, n_ancilla + unitary_circuit.n_qubits))
    target_qubits = list(target_qubits)
    qc = QuantumCircuit(n_ancilla + unitary_circuit.n_qubits, name="QPE")
    anc = list(range(n_ancilla))
    qc.h(anc)
    for k, a in enumerate(anc):
        repeats = 2**k
        for _ in range(repeats):
            for inst in unitary_circuit.instructions:
                copied = qc.append(inst.name, [target_qubits[q] for q in inst.qubits], params=list(inst.params))
                copied.control(a)
    inverse_qft(qc, anc)
    return qc


def phases_from_counts(counts, n_ancilla: int) -> PhaseEstimationResult:
    total = sum(counts.values())
    bitstrings = sorted(counts, key=counts.get, reverse=True)
    probs = np.array([counts[b] / total for b in bitstrings], dtype=float)
    phases = np.array([int(b, 2) / (2**n_ancilla) for b in bitstrings], dtype=float)
    return PhaseEstimationResult(phases=phases, probabilities=probs, bitstrings=bitstrings)


def exact_phase_estimation_from_unitary(unitary: np.ndarray, input_state: np.ndarray, n_ancilla: int, energy_scale: Optional[float] = None) -> PhaseEstimationResult:
    """Classical reference for QPE output probabilities."""

    vals, vecs = np.linalg.eig(unitary)
    phases = (np.angle(vals) / (2 * np.pi)) % 1.0
    overlaps = np.abs(vecs.conj().T @ input_state) ** 2
    grid = np.arange(2**n_ancilla) / (2**n_ancilla)
    probs = np.zeros_like(grid, dtype=float)
    for phase, weight in zip(phases, overlaps):
        idx = int(np.round(phase * 2**n_ancilla)) % (2**n_ancilla)
        probs[idx] += weight.real
    energies = None
    if energy_scale is not None:
        energies = 2 * np.pi * grid / energy_scale
    return PhaseEstimationResult(phases=grid, probabilities=probs, energies=energies, bitstrings=[format(i, f"0{n_ancilla}b") for i in range(2**n_ancilla)])


def iterative_phase_estimation_step(expectation_cos: float, expectation_sin: float) -> float:
    """Return a phase estimate from Hadamard-test cosine/sine estimates."""

    return float(np.arctan2(expectation_sin, expectation_cos) / (2 * np.pi) % 1.0)
