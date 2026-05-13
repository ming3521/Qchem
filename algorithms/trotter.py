"""Hamiltonian simulation and Trotterization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backends.core import QuantumCircuit

PauliString = Tuple[Tuple[int, str], ...]
PauliTerm = Tuple[PauliString, complex]


@dataclass
class TrotterStepInfo:
    order: int
    time: float
    n_steps: int
    n_terms: int
    gate_count_estimate: int


def normalize_pauli_terms(hamiltonian) -> List[PauliTerm]:
    """Normalize common Hamiltonian representations to ``[(term, coeff), ...]``."""

    if hasattr(hamiltonian, "terms"):
        return [(tuple(term), complex(coeff)) for term, coeff in hamiltonian.terms.items()]
    if isinstance(hamiltonian, Mapping):
        return [(tuple(term), complex(coeff)) for term, coeff in hamiltonian.items()]
    out = []
    for item in hamiltonian:
        if len(item) != 2:
            raise ValueError("Hamiltonian entries must be (pauli_string, coefficient)")
        term, coeff = item
        out.append((tuple(term), complex(coeff)))
    return out


def append_pauli_evolution(qc: QuantumCircuit, pauli_string: PauliString, angle):
    """Append exp(-i angle P) for a Pauli string P."""

    if not pauli_string:
        qc.global_phase(-angle)
        return
    qubits = [int(q) for q, _ in pauli_string]
    ops = [str(p).upper() for _, p in pauli_string]
    for q, op in zip(qubits, ops):
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.rx(q, np.pi / 2)
        elif op == "Z":
            pass
        else:
            raise ValueError(f"Unsupported Pauli operator '{op}'")
    for a, b in zip(qubits[:-1], qubits[1:]):
        qc.cx(a, b)
    qc.rz(qubits[-1], 2.0 * angle)
    for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
        qc.cx(a, b)
    for q, op in reversed(list(zip(qubits, ops))):
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.rx(q, -np.pi / 2)


def first_order_trotter_circuit(hamiltonian, time: float, n_steps: int, n_qubits: Optional[int] = None) -> QuantumCircuit:
    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = 0
        for term, _ in terms:
            for q, _p in term:
                n_qubits = max(n_qubits, int(q) + 1)
    qc = QuantumCircuit(n_qubits, name="FirstOrderTrotter")
    dt = float(time) / int(n_steps)
    for _ in range(int(n_steps)):
        for term, coeff in terms:
            if abs(coeff.imag) > 1e-10:
                raise ValueError("Trotter evolution expects real Hamiltonian coefficients")
            append_pauli_evolution(qc, term, dt * coeff.real)
    return qc


def second_order_trotter_circuit(hamiltonian, time: float, n_steps: int, n_qubits: Optional[int] = None) -> QuantumCircuit:
    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = max((int(q) + 1 for term, _ in terms for q, _p in term), default=0)
    qc = QuantumCircuit(n_qubits, name="SecondOrderTrotter")
    dt = float(time) / int(n_steps)
    for _ in range(int(n_steps)):
        for term, coeff in terms:
            append_pauli_evolution(qc, term, 0.5 * dt * coeff.real)
        for term, coeff in reversed(terms):
            append_pauli_evolution(qc, term, 0.5 * dt * coeff.real)
    return qc


def trotter_info(hamiltonian, time: float, n_steps: int, order: int = 1) -> TrotterStepInfo:
    terms = normalize_pauli_terms(hamiltonian)
    estimated = 0
    for term, _ in terms:
        k = len(term)
        estimated += 1 if k == 0 else 2 * max(k - 1, 0) + 1 + 2 * sum(p != "Z" for _, p in term)
    multiplier = n_steps if order == 1 else 2 * n_steps
    return TrotterStepInfo(order=order, time=float(time), n_steps=int(n_steps), n_terms=len(terms), gate_count_estimate=estimated * multiplier)
