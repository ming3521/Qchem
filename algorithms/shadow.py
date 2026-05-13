"""Classical-shadow measurement and estimator utilities.

The module implements the standard local-random-Pauli classical shadow protocol:
for each snapshot, each qubit is measured independently in X, Y, or Z.  A Pauli
observable is estimated with the inverse single-qubit measurement channel, so a
weight-w Pauli string has per-snapshot contribution 0 unless all non-identity
positions were measured in the matching bases, and otherwise contributes
``3**w * product(eigenvalues)``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from backends.core import QuantumCircuit


_AXIS_TO_CHAR = {0: "X", 1: "Y", 2: "Z"}
_CHAR_TO_AXIS = {"X": 0, "Y": 1, "Z": 2}
_VALID_PAULI = {"I", "X", "Y", "Z"}

PauliString = Tuple[Tuple[int, str], ...]
PauliTerm = Tuple[PauliString, complex]


@dataclass(frozen=True)
class ShadowSnapshot:
    """One local-Pauli classical-shadow snapshot.

    ``bits[q]`` is the computational-basis bit measured after rotating qubit ``q``
    from its chosen measurement basis to Z.  ``axes[q]`` is encoded as
    0 -> X, 1 -> Y, 2 -> Z.
    """

    bits: Tuple[int, ...]
    axes: Tuple[int, ...]


@dataclass
class ObservableEstimate:
    """Small container for observable estimates."""

    value: complex
    n_snapshots: int
    pauli_string: Optional[str] = None

    @property
    def real_if_close(self):
        return np.real_if_close(self.value)


def _as_rng(seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _validate_n_snapshots(n_snapshots: int) -> int:
    if not isinstance(n_snapshots, (int, np.integer)):
        raise TypeError("n_snapshots must be an integer")
    n = int(n_snapshots)
    if n <= 0:
        raise ValueError("n_snapshots must be positive")
    return n


def _normalize_pauli_string(pauli_string: str, n_qubits: Optional[int] = None) -> str:
    if not isinstance(pauli_string, str):
        raise TypeError("Pauli observable must be a string such as 'ZI' or 'XX'")
    pauli = pauli_string.upper().replace(" ", "")
    if not pauli:
        raise ValueError("Pauli observable string cannot be empty")
    bad = sorted(set(pauli) - _VALID_PAULI)
    if bad:
        raise ValueError(f"Unsupported Pauli characters: {bad}; allowed are I, X, Y, Z")
    if n_qubits is not None and len(pauli) != int(n_qubits):
        raise ValueError(f"Pauli string length {len(pauli)} does not match n_qubits={n_qubits}")
    return pauli


def _term_to_pauli_string(term: PauliString, n_qubits: int) -> str:
    chars = ["I"] * int(n_qubits)
    seen = set()
    for q, p in term:
        q = int(q)
        p = str(p).upper()
        if q < 0 or q >= n_qubits:
            raise ValueError(f"Pauli term qubit {q} is outside [0, {n_qubits - 1}]")
        if q in seen:
            raise ValueError(f"Duplicate qubit {q} in Pauli term {term}")
        if p not in _VALID_PAULI:
            raise ValueError(f"Unsupported Pauli operator '{p}'")
        chars[q] = p
        seen.add(q)
    return "".join(chars)


def _string_to_term(pauli_string: str) -> PauliString:
    pauli = _normalize_pauli_string(pauli_string)
    return tuple((q, p) for q, p in enumerate(pauli) if p != "I")


def _looks_like_number(x) -> bool:
    try:
        complex(x)
        return not isinstance(x, str)
    except Exception:
        return False


def normalize_pauli_terms(hamiltonian) -> List[PauliTerm]:
    """Normalize common Hamiltonian representations to ``[(term, coeff), ...]``.

    Supported examples include:

    - ``[(coeff, "ZI"), (coeff, "XX")]`` -- the library's standard format.
    - ``{"ZI": coeff, "XX": coeff}``.
    - OpenFermion-like ``{((0, "Z"), (1, "I")): coeff}`` or objects with
      a ``.terms`` dictionary.
    - Legacy ``[(term, coeff)]`` where ``term`` is ``((q, pauli), ...)``.
    """

    if hasattr(hamiltonian, "terms"):
        return [(tuple(term), complex(coeff)) for term, coeff in hamiltonian.terms.items()]

    if isinstance(hamiltonian, Mapping):
        out: List[PauliTerm] = []
        for key, coeff in hamiltonian.items():
            if isinstance(key, str):
                out.append((_string_to_term(key), complex(coeff)))
            else:
                out.append((tuple(key), complex(coeff)))
        return out

    if isinstance(hamiltonian, str):
        return [(_string_to_term(hamiltonian), 1.0 + 0.0j)]

    out: List[PauliTerm] = []
    for item in hamiltonian:
        if isinstance(item, str):
            out.append((_string_to_term(item), 1.0 + 0.0j))
            continue
        if len(item) != 2:
            raise ValueError("Hamiltonian entries must be (coefficient, pauli_string) or (term, coefficient)")
        first, second = item
        if _looks_like_number(first) and isinstance(second, str):
            out.append((_string_to_term(second), complex(first)))
        elif isinstance(first, str) and _looks_like_number(second):
            out.append((_string_to_term(first), complex(second)))
        else:
            out.append((tuple(first), complex(second)))
    return out


def _single_snapshot_pauli_value(bits: Sequence[int], axes: Sequence[int], pauli_string: str) -> float:
    value = 1.0
    for bit, measured_axis, target in zip(bits, axes, pauli_string):
        if target == "I":
            continue
        target_axis = _CHAR_TO_AXIS[target]
        if int(measured_axis) != target_axis:
            return 0.0
        value *= 3.0 * (1.0 if int(bit) == 0 else -1.0)
    return float(value)


def _parse_single_shot_counts(counts: Mapping[str, int], n_qubits: int) -> Tuple[int, ...]:
    if not counts:
        raise ValueError("Backend returned an empty counts dictionary")
    positive = [(str(bitstring), int(count)) for bitstring, count in counts.items() if int(count) > 0]
    if not positive:
        raise ValueError("Backend returned no positive-count outcomes")
    total = sum(count for _, count in positive)
    if total != 1:
        raise ValueError(
            "collect() expects backend.run_sampling(..., shots=1) to return exactly one shot; "
            f"got total count {total}"
        )
    bitstring = positive[0][0]
    if len(bitstring) != n_qubits:
        raise ValueError(f"Measured bitstring length {len(bitstring)} does not match n_qubits={n_qubits}")
    if any(c not in "01" for c in bitstring):
        raise ValueError(f"Invalid measured bitstring '{bitstring}'")
    return tuple(int(c) for c in bitstring)


class ClassicalShadow:
    """Collect and use local-random-Pauli classical-shadow snapshots.

    Parameters
    ----------
    backend:
        Backend object exposing ``run_sampling(circuit, shots, measure_qubits)``.
    circuit:
        State-preparation circuit.
    n_qubits:
        Number of qubits to measure.  Defaults to ``circuit.n_qubits``.
    seed:
        Optional seed or ``np.random.Generator`` for reproducible basis choices.
    """

    def __init__(
        self,
        backend,
        circuit: QuantumCircuit,
        n_qubits: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.backend = backend
        self.circuit = circuit
        inferred = int(getattr(circuit, "n_qubits"))
        self.n_qubits = inferred if n_qubits is None else int(n_qubits)
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.n_qubits != inferred:
            raise ValueError(
                f"n_qubits={self.n_qubits} does not match circuit.n_qubits={inferred}. "
                "This implementation measures all qubits of the state-preparation circuit."
            )
        self.rng = _as_rng(seed)
        self.snapshots: List[ShadowSnapshot] = []

    def _append_rotation(self, qc: QuantumCircuit, axes: Sequence[int]):
        """Append basis rotations so X/Y/Z measurements are performed in Z basis."""

        if len(axes) != self.n_qubits:
            raise ValueError(f"axes length {len(axes)} does not match n_qubits={self.n_qubits}")
        for q, axis in enumerate(axes):
            axis = int(axis)
            if axis == 0:  # X measurement: H^†d Z H = X
                qc.h(q)
            elif axis == 1:  # Y measurement: (H S^†d)^†d Z (H S^†d) = Y
                qc.sdg(q)
                qc.h(q)
            elif axis == 2:  # Z measurement
                pass
            else:
                raise ValueError("Measurement axes must be encoded as 0:X, 1:Y, 2:Z")
        return qc

    def collect(
        self,
        n_snapshots: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
        append: bool = False,
    ) -> List[ShadowSnapshot]:
        """Collect classical-shadow snapshots.

        The backend is called with ``shots=1`` for each random Pauli basis.  The
        returned list is also stored in ``self.snapshots``.
        """

        n = _validate_n_snapshots(n_snapshots)
        rng = self.rng if seed is None else _as_rng(seed)
        if not append:
            self.snapshots = []

        for _ in range(n):
            axes = tuple(int(x) for x in rng.integers(0, 3, size=self.n_qubits))
            run_qc = copy.deepcopy(self.circuit)
            self._append_rotation(run_qc, axes)
            counts = self.backend.run_sampling(run_qc, shots=1, measure_qubits=list(range(self.n_qubits)))
            bits = _parse_single_shot_counts(counts, self.n_qubits)
            self.snapshots.append(ShadowSnapshot(bits=bits, axes=axes))

        return self.snapshots

    def estimate_pauli_string(self, pauli_string: str) -> float:
        """Estimate ``<pauli_string>`` from collected snapshots."""

        if not self.snapshots:
            raise ValueError("Please call collect() before estimating observables")
        pauli = _normalize_pauli_string(pauli_string, self.n_qubits)
        values = [_single_snapshot_pauli_value(s.bits, s.axes, pauli) for s in self.snapshots]
        return float(np.mean(values))

    def estimate_observable(self, observable_pauli_string: str) -> float:
        """Backward-compatible alias for ``estimate_pauli_string``."""

        return self.estimate_pauli_string(observable_pauli_string)

    def estimate_hamiltonian(self, hamiltonian) -> complex:
        """Estimate a Hamiltonian expectation value from Pauli-term input."""

        if not self.snapshots:
            raise ValueError("Please call collect() before estimating observables")
        total = 0.0 + 0.0j
        for term, coeff in normalize_pauli_terms(hamiltonian):
            pauli = _term_to_pauli_string(term, self.n_qubits)
            if set(pauli) == {"I"}:
                total += coeff
            else:
                total += coeff * self.estimate_pauli_string(pauli)
        return complex(total)

    def estimate_density_matrix(self, max_qubits: int = 10) -> np.ndarray:
        """Reconstruct the average shadow density matrix.

        This is useful for diagnostics and small systems only.  The matrix size is
        ``2**n_qubits`` by ``2**n_qubits``.
        """

        if not self.snapshots:
            raise ValueError("Please call collect() before estimating the density matrix")
        if self.n_qubits > max_qubits:
            raise ValueError(
                f"Density-matrix reconstruction is exponential; n_qubits={self.n_qubits} exceeds max_qubits={max_qubits}"
            )
        matrices = [snapshot_matrix(snapshot) for snapshot in self.snapshots]
        return np.mean(matrices, axis=0)


def snapshot_matrix(snapshot: Union[ShadowSnapshot, Tuple[Sequence[int], Sequence[int]]]) -> np.ndarray:
    """Return the tensor-product inverse-channel matrix for one snapshot."""

    if isinstance(snapshot, ShadowSnapshot):
        bits, axes = snapshot.bits, snapshot.axes
    else:
        bits, axes = snapshot
    if len(bits) != len(axes):
        raise ValueError("Snapshot bits and axes must have the same length")

    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    paulis = [X, Y, Z]

    mat = np.array([[1.0 + 0.0j]])
    for bit, axis in zip(bits, axes):
        axis = int(axis)
        if axis not in (0, 1, 2):
            raise ValueError("Measurement axes must be encoded as 0:X, 1:Y, 2:Z")
        eig = 1.0 if int(bit) == 0 else -1.0
        local = 0.5 * (I + 3.0 * eig * paulis[axis])
        mat = np.kron(mat, local)
    return mat


def estimate_pauli_from_snapshots(
    snapshots: Sequence[Union[ShadowSnapshot, Tuple[Sequence[int], Sequence[int]]]],
    pauli_string: str,
) -> float:
    """Functional estimator for a Pauli string from explicit snapshots."""

    if not snapshots:
        raise ValueError("snapshots cannot be empty")
    first = snapshots[0]
    n_qubits = len(first.bits if isinstance(first, ShadowSnapshot) else first[0])
    pauli = _normalize_pauli_string(pauli_string, n_qubits)
    values = []
    for snapshot in snapshots:
        if isinstance(snapshot, ShadowSnapshot):
            bits, axes = snapshot.bits, snapshot.axes
        else:
            bits, axes = snapshot
        if len(bits) != n_qubits or len(axes) != n_qubits:
            raise ValueError("All snapshots must have the same number of qubits")
        values.append(_single_snapshot_pauli_value(bits, axes, pauli))
    return float(np.mean(values))


__all__ = [
    "ClassicalShadow",
    "ShadowSnapshot",
    "ObservableEstimate",
    "normalize_pauli_terms",
    "estimate_pauli_from_snapshots",
    "snapshot_matrix",
]
