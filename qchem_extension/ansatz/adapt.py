"""ADAPT-VQE style adaptive ansatz tools.

ADAPT is not a single fixed circuit.  It is a workflow that grows a circuit by
screening an operator pool and appending the operator with the largest energy
gradient.  The class below stores the selected operators and emits a circuit with
one trainable parameter per selected generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ansatz.base import Ansatz
from backends.core import QuantumCircuit

PauliTerm = Tuple[Tuple[Tuple[int, str], ...], float]
Operator = List[PauliTerm]


def _as_tensor(params):
    if isinstance(params, torch.Tensor):
        return params
    return torch.tensor(params, dtype=torch.float64)


@dataclass
class AdaptOperator:
    """A selected anti-Hermitian generator represented by Pauli terms."""

    label: str
    terms: Operator
    source: Optional[Tuple[int, ...]] = None
    target: Optional[Tuple[int, ...]] = None


class ADAPTAnsatz(Ansatz):
    """Adaptive ansatz that stores selected ADAPT operators.

    Parameters are ordered exactly as operators were added.  The ansatz prepares a
    Hartree-Fock reference state and then applies each generator with a first-order
    product formula.  The Pauli evolution routine mirrors the implementation used
    by UCCSD so that both circuit families compile to the same gate vocabulary.
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int = 0,
        selected_operators: Optional[Sequence[AdaptOperator]] = None,
        trotter_steps: int = 1,
        name: str = "ADAPTAnsatz",
    ):
        self.n_electrons = int(n_electrons)
        self.selected_operators: List[AdaptOperator] = list(selected_operators or [])
        self.trotter_steps = int(trotter_steps)
        self.name = name
        super().__init__(n_qubits, len(self.selected_operators))

    def add_operator(self, operator: AdaptOperator):
        self.selected_operators.append(operator)
        self.n_params = len(self.selected_operators)
        return self

    def extend(self, operators: Sequence[AdaptOperator]):
        for op in operators:
            self.add_operator(op)
        return self

    def copy(self) -> "ADAPTAnsatz":
        return ADAPTAnsatz(
            self.n_qubits,
            self.n_electrons,
            selected_operators=list(self.selected_operators),
            trotter_steps=self.trotter_steps,
            name=self.name,
        )

    def _prepare_hf(self, qc: QuantumCircuit):
        for q in range(min(self.n_electrons, self.n_qubits)):
            qc.x(q)

    @staticmethod
    def _append_pauli_evolution(qc: QuantumCircuit, pauli_string: Tuple[Tuple[int, str], ...], angle):
        if not pauli_string:
            return
        qubits = [int(idx) for idx, _ in pauli_string]
        ops = [str(op).upper() for _, op in pauli_string]
        for q, op in zip(qubits, ops):
            if op == "X":
                qc.h(q)
            elif op == "Y":
                qc.rx(q, np.pi / 2)
            elif op == "Z":
                pass
            else:
                raise ValueError(f"Unsupported Pauli operator '{op}'")
        for left, right in zip(qubits[:-1], qubits[1:]):
            qc.cx(left, right)
        qc.rz(qubits[-1], 2.0 * angle)
        for left, right in reversed(list(zip(qubits[:-1], qubits[1:]))):
            qc.cx(left, right)
        for q, op in reversed(list(zip(qubits, ops))):
            if op == "X":
                qc.h(q)
            elif op == "Y":
                qc.rx(q, -np.pi / 2)

    def forward(self, params) -> QuantumCircuit:
        params = _as_tensor(params)
        if params.numel() != self.n_params:
            raise ValueError(f"ADAPTAnsatz expected {self.n_params} parameters, got {params.numel()}")
        qc = QuantumCircuit(self.n_qubits, name=self.name)
        self._prepare_hf(qc)
        for _ in range(self.trotter_steps):
            for theta, op in zip(params, self.selected_operators):
                for pauli_string, coeff in op.terms:
                    self._append_pauli_evolution(qc, pauli_string, theta * coeff)
        return qc

    def zero_parameters(self) -> torch.Tensor:
        return torch.zeros(self.n_params, dtype=torch.float64)


def _jw_single_excitation(i: int, a: int) -> Operator:
    """Return a compact JW-like anti-Hermitian single excitation generator.

    The exact Jordan-Wigner image of a_i^dagger a_j - h.c. contains X/Y strings
    and parity Z strings.  This helper creates the same physically meaningful
    rotation pattern without requiring OpenFermion at import time.
    """

    lo, hi = sorted((i, a))
    z_string = tuple((q, "Z") for q in range(lo + 1, hi))
    return [
        (tuple([(lo, "X"), *z_string, (hi, "Y")]), 0.5),
        (tuple([(lo, "Y"), *z_string, (hi, "X")]), -0.5),
    ]


def _jw_pair_excitation(i: int, j: int, a: int, b: int) -> Operator:
    """Return an approximate spin-adapted double-excitation Pauli generator."""

    qubits = [i, j, a, b]
    patterns = [
        ("XXXY", 1.0 / 8.0),
        ("XXYX", 1.0 / 8.0),
        ("XYXX", -1.0 / 8.0),
        ("YXXX", -1.0 / 8.0),
        ("YYXY", 1.0 / 8.0),
        ("YYYX", 1.0 / 8.0),
        ("YXYX", -1.0 / 8.0),
        ("XYYY", -1.0 / 8.0),
    ]
    out: Operator = []
    for pattern, coeff in patterns:
        out.append((tuple((q, p) for q, p in zip(qubits, pattern)), coeff))
    return out


def build_uccsd_operator_pool(n_qubits: int, n_electrons: int, include_singles: bool = True, include_doubles: bool = True) -> List[AdaptOperator]:
    """Build a singles/doubles pool for ADAPT screening."""

    occ = list(range(n_electrons))
    vir = list(range(n_electrons, n_qubits))
    pool: List[AdaptOperator] = []
    if include_singles:
        for i in occ:
            for a in vir:
                pool.append(AdaptOperator(label=f"S({i}->{a})", terms=_jw_single_excitation(i, a), source=(i,), target=(a,)))
    if include_doubles:
        for x, i in enumerate(occ):
            for j in occ[x + 1 :]:
                for y, a in enumerate(vir):
                    for b in vir[y + 1 :]:
                        pool.append(
                            AdaptOperator(
                                label=f"D({i},{j}->{a},{b})",
                                terms=_jw_pair_excitation(i, j, a, b),
                                source=(i, j),
                                target=(a, b),
                            )
                        )
    return pool


def build_pair_operator_pool(n_qubits: int, n_electrons: int) -> List[AdaptOperator]:
    """Build a compact pair-only ADAPT pool."""

    if n_qubits % 2 != 0 or n_electrons % 2 != 0:
        raise ValueError("Pair pool requires even n_qubits and n_electrons")
    n_occ_pairs = n_electrons // 2
    n_pairs = n_qubits // 2
    pool: List[AdaptOperator] = []
    for p in range(n_occ_pairs):
        for q in range(n_occ_pairs, n_pairs):
            i, j = 2 * p, 2 * p + 1
            a, b = 2 * q, 2 * q + 1
            pool.append(AdaptOperator(label=f"P({p}->{q})", terms=_jw_pair_excitation(i, j, a, b), source=(i, j), target=(a, b)))
    return pool


def finite_difference_gradient(
    energy_fn: Callable[[torch.Tensor], float],
    params: torch.Tensor,
    operator_index: int,
    epsilon: float = 1e-4,
) -> float:
    """Estimate the gradient for a newly appended ADAPT operator.

    This helper is intentionally optimizer-agnostic.  A caller can append a trial
    operator with zero parameter, evaluate plus/minus shifts, then choose the
    largest absolute gradient.
    """

    shift = torch.zeros_like(params)
    shift[operator_index] = epsilon
    e_plus = float(energy_fn(params + shift))
    e_minus = float(energy_fn(params - shift))
    return (e_plus - e_minus) / (2.0 * epsilon)


def select_largest_gradient(gradients: Sequence[float], pool: Sequence[AdaptOperator], threshold: float = 1e-4):
    if len(gradients) != len(pool):
        raise ValueError("Gradient list and operator pool must have the same length")
    if not gradients:
        return None, 0.0, True
    idx = int(np.argmax(np.abs(np.asarray(gradients, dtype=float))))
    max_grad = float(gradients[idx])
    converged = abs(max_grad) < threshold
    return pool[idx], max_grad, converged
