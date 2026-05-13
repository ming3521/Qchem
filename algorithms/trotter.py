"""Hamiltonian simulation and Trotterization utilities.

This module builds backend-independent HYQ ``QuantumCircuit`` objects for
first- and second-order product-formula time evolution,

    U(t) ~= exp(-i H t).

Hamiltonians are normalized to the same Pauli-term convention used by the
algorithm helpers in this library.  In particular, the standard library format

    [(coeff, "ZI"), (coeff, "XX")]

is supported, while older OpenFermion-like ``[((q, "P"), ...), coeff]`` terms
remain accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import List, Mapping, Optional, Tuple

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


def _is_scalar(value) -> bool:
    if isinstance(value, Number):
        return True
    arr = np.asarray(value)
    return arr.shape == () and np.issubdtype(arr.dtype, np.number)


def _coerce_coeff(value) -> complex:
    try:
        return complex(value)
    except Exception as exc:
        raise TypeError(f"Invalid Hamiltonian coefficient {value!r}") from exc


def _term_from_pauli_string(pauli_string: str) -> PauliString:
    """Convert a dense Pauli string such as ``'ZIX'`` to indexed terms.

    Identity characters are retained here so that qubit-count inference still
    works for strings with trailing identities, e.g. ``'ZI'`` implies two qubits.
    ``append_pauli_evolution`` filters identities before adding gates.
    """

    text = str(pauli_string).strip().upper()
    if text in {"", "1"}:
        return tuple()
    out = []
    for q, pauli in enumerate(text):
        if pauli not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"Unsupported Pauli operator '{pauli}' in string {pauli_string!r}")
        out.append((q, pauli))
    return tuple(out)


def _term_from_indexed_ops(term) -> PauliString:
    """Normalize an OpenFermion-like indexed Pauli term."""

    if term is None:
        return tuple()
    if isinstance(term, str):
        return _term_from_pauli_string(term)

    pairs = list(term)
    if len(pairs) == 0:
        return tuple()

    out = []
    seen = set()
    for item in pairs:
        if len(item) != 2:
            raise ValueError(f"Pauli term entries must be (qubit, pauli), got {item!r}")
        q, pauli = item
        q = int(q)
        if q < 0:
            raise ValueError(f"Qubit index must be non-negative, got {q}")
        p = str(pauli).strip().upper()
        if p not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"Unsupported Pauli operator '{pauli}'")
        if q in seen:
            raise ValueError(f"Duplicate qubit index {q} in Pauli string {term!r}")
        seen.add(q)
        out.append((q, p))
    return tuple(sorted(out, key=lambda x: x[0]))


def normalize_pauli_terms(hamiltonian) -> List[PauliTerm]:
    """Normalize common Hamiltonian representations to ``[(term, coeff), ...]``.

    Accepted formats include:

    - ``[(coeff, "ZI"), (coeff, "XX")]``  (HYQ standard format)
    - ``{"ZI": coeff, "XX": coeff}``
    - ``[((0, "Z"), (1, "I")), coeff]`` style entries
    - OpenFermion-like objects with a ``.terms`` mapping
    """

    if hamiltonian is None:
        raise ValueError("Hamiltonian cannot be None")

    if hasattr(hamiltonian, "terms"):
        iterable = list(hamiltonian.terms.items())
    elif isinstance(hamiltonian, Mapping):
        iterable = list(hamiltonian.items())
    else:
        iterable = list(hamiltonian)

    out: List[PauliTerm] = []
    for item in iterable:
        if len(item) != 2:
            raise ValueError("Hamiltonian entries must contain exactly two fields")

        first, second = item

        # HYQ standard: (coefficient, "PauliString")
        if _is_scalar(first) and isinstance(second, str):
            coeff = _coerce_coeff(first)
            term = _term_from_pauli_string(second)
        # Convenient dict/list alternative: ("PauliString", coefficient)
        elif isinstance(first, str) and _is_scalar(second):
            term = _term_from_pauli_string(first)
            coeff = _coerce_coeff(second)
        # OpenFermion-like: (indexed_term, coefficient)
        else:
            term = _term_from_indexed_ops(first)
            coeff = _coerce_coeff(second)

        out.append((term, coeff))
    return out


def infer_n_qubits(terms: List[PauliTerm]) -> int:
    n_qubits = 0
    for term, _ in terms:
        for q, _pauli in term:
            n_qubits = max(n_qubits, int(q) + 1)
    return n_qubits


def _validate_n_steps(n_steps: int) -> int:
    try:
        n_steps_int = int(n_steps)
    except Exception as exc:
        raise TypeError("n_steps must be a positive integer") from exc
    if n_steps_int != n_steps:
        raise ValueError("n_steps must be an integer")
    if n_steps_int <= 0:
        raise ValueError("n_steps must be positive")
    return n_steps_int


def _validate_time(time: float) -> float:
    time = float(time)
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    return time


def _real_coeff(coeff: complex, tol: float = 1e-10) -> float:
    coeff = complex(coeff)
    if abs(coeff.imag) > tol:
        raise ValueError(
            "Trotter evolution expects Hermitian Pauli Hamiltonians with real coefficients; "
            f"got coefficient {coeff!r}"
        )
    return float(coeff.real)


def append_pauli_evolution(qc: QuantumCircuit, pauli_string: PauliString, angle):
    """Append gates implementing ``exp(-i angle P)`` for a Pauli string ``P``.

    ``QuantumCircuit.rz(theta)`` is ``exp(-i theta Z / 2)``, hence the central
    rotation angle is ``2 * angle``.
    """

    angle = float(angle)
    term = _term_from_indexed_ops(pauli_string)
    active = [(int(q), str(p).upper()) for q, p in term if str(p).upper() != "I"]

    if not active:
        qc.global_phase(-angle)
        return qc

    qubits = [q for q, _ in active]
    ops = [p for _, p in active]

    for q, op in zip(qubits, ops):
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.rx(q, np.pi / 2)
        elif op == "Z":
            pass
        else:  # pragma: no cover - guarded by normalization
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

    return qc


def _validate_terms_real(terms: List[PauliTerm]) -> List[Tuple[PauliString, float]]:
    return [(term, _real_coeff(coeff)) for term, coeff in terms]


def first_order_trotter_circuit(
    hamiltonian,
    time: float,
    n_steps: int,
    n_qubits: Optional[int] = None,
) -> QuantumCircuit:
    """Build first-order Lie-Trotter evolution for ``exp(-i H time)``."""

    terms = normalize_pauli_terms(hamiltonian)
    real_terms = _validate_terms_real(terms)
    if n_qubits is None:
        n_qubits = infer_n_qubits(terms)
    n_qubits = int(n_qubits)
    if n_qubits < infer_n_qubits(terms):
        raise ValueError("n_qubits is smaller than required by Hamiltonian terms")

    steps = _validate_n_steps(n_steps)
    dt = _validate_time(time) / steps

    qc = QuantumCircuit(n_qubits, name="FirstOrderTrotter")
    for _ in range(steps):
        for term, coeff in real_terms:
            append_pauli_evolution(qc, term, dt * coeff)
    return qc


def second_order_trotter_circuit(
    hamiltonian,
    time: float,
    n_steps: int,
    n_qubits: Optional[int] = None,
) -> QuantumCircuit:
    """Build second-order symmetric Strang Trotter evolution for ``exp(-i H time)``."""

    terms = normalize_pauli_terms(hamiltonian)
    real_terms = _validate_terms_real(terms)
    if n_qubits is None:
        n_qubits = infer_n_qubits(terms)
    n_qubits = int(n_qubits)
    if n_qubits < infer_n_qubits(terms):
        raise ValueError("n_qubits is smaller than required by Hamiltonian terms")

    steps = _validate_n_steps(n_steps)
    dt = _validate_time(time) / steps

    qc = QuantumCircuit(n_qubits, name="SecondOrderTrotter")
    for _ in range(steps):
        for term, coeff in real_terms:
            append_pauli_evolution(qc, term, 0.5 * dt * coeff)
        for term, coeff in reversed(real_terms):
            append_pauli_evolution(qc, term, 0.5 * dt * coeff)
    return qc


def trotter_circuit(
    hamiltonian,
    time: float,
    n_steps: int,
    order: int = 1,
    n_qubits: Optional[int] = None,
) -> QuantumCircuit:
    """Dispatch to a first- or second-order Trotter circuit builder."""

    if order == 1:
        return first_order_trotter_circuit(hamiltonian, time, n_steps, n_qubits=n_qubits)
    if order == 2:
        return second_order_trotter_circuit(hamiltonian, time, n_steps, n_qubits=n_qubits)
    raise ValueError("Only first- and second-order Trotter formulas are supported")


def _gate_count_for_term(term: PauliString) -> int:
    active = [(q, p) for q, p in term if str(p).upper() != "I"]
    k = len(active)
    if k == 0:
        return 1  # one global phase instruction
    basis_changes = 2 * sum(str(p).upper() != "Z" for _, p in active)
    cnot_ladder = 2 * max(k - 1, 0)
    central_rz = 1
    return basis_changes + cnot_ladder + central_rz


def trotter_info(hamiltonian, time: float, n_steps: int, order: int = 1) -> TrotterStepInfo:
    """Return a simple gate-count estimate for the generated Trotter circuit."""

    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2")
    steps = _validate_n_steps(n_steps)
    time = _validate_time(time)
    terms = normalize_pauli_terms(hamiltonian)
    _validate_terms_real(terms)

    per_term_estimate = sum(_gate_count_for_term(term) for term, _ in terms)
    multiplier = steps if order == 1 else 2 * steps
    return TrotterStepInfo(
        order=order,
        time=time,
        n_steps=steps,
        n_terms=len(terms),
        gate_count_estimate=per_term_estimate * multiplier,
    )
