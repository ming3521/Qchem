"""
Dense state-vector utilities shared by optional HYQ-ALG-LIB backends.

The original project stores circuits in a lightweight backend-independent
``QuantumCircuit`` object.  Each instruction contains a gate name, target qubits,
optional parameters, optional control qubits, and optional dense matrix.  The
helpers in this file translate that representation into a pure NumPy state-vector
simulation.  Optional backends such as Cirq, Qulacs, and QuTiP can use these
functions as a correctness reference and as a fallback when a native gate is not
available.

The code intentionally avoids importing Qiskit/PennyLane/TensorCircuit.  It is
therefore safe to import even when only NumPy and Torch are installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # torch is already used by the original backend API, but keep import optional.
    import torch
except Exception:  # pragma: no cover - import guard for documentation builds
    torch = None  # type: ignore


_EPS = 1e-12


def _to_float(value) -> float:
    """Convert Python/NumPy/Torch scalar-like values to a plain float."""

    if torch is not None and hasattr(value, "detach"):
        return float(value.detach().cpu().numpy())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _complex_array(matrix) -> np.ndarray:
    """Return a complex128 NumPy array without modifying the input object."""

    if torch is not None and hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.complex128)


def _identity() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def _x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _y() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)


def _z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _h() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


def _s() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)


def _sdg() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)


def _t() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)


def _tdg() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(-1.0j * np.pi / 4.0)]], dtype=np.complex128)


def _sx() -> np.ndarray:
    return 0.5 * np.array(
        [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=np.complex128
    )


def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=np.complex128
    )


def _phase(theta: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * theta)]], dtype=np.complex128)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array(
        [
            [c, -np.exp(1.0j * lam) * s],
            [np.exp(1.0j * phi) * s, np.exp(1.0j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def _swap() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )


def _iswap() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )


def _rxx(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * np.eye(4, dtype=np.complex128) - 1j * math.sin(theta / 2.0) * np.kron(_x(), _x())


def _ryy(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * np.eye(4, dtype=np.complex128) - 1j * math.sin(theta / 2.0) * np.kron(_y(), _y())


def _rzz(theta: float) -> np.ndarray:
    return np.diag(
        [np.exp(-0.5j * theta), np.exp(0.5j * theta), np.exp(0.5j * theta), np.exp(-0.5j * theta)]
    ).astype(np.complex128)


def _controlled(base: np.ndarray, ctrl_value: int = 1) -> np.ndarray:
    """Return a one-control version of a k-qubit base unitary."""

    dim = base.shape[0]
    result = np.eye(2 * dim, dtype=np.complex128)
    start = dim if ctrl_value == 1 else 0
    result[start : start + dim, start : start + dim] = base
    return result


def gate_matrix(name: str, params: Optional[Sequence] = None) -> Optional[np.ndarray]:
    """Return a dense unitary matrix for a HYQ gate name.

    Unsupported names return ``None`` so that native backends can choose their own
    implementation or skip/raise a useful error.
    """

    params = list(params or [])
    key = name.lower()
    single_no_param = {
        "id": _identity,
        "i": _identity,
        "x": _x,
        "y": _y,
        "z": _z,
        "h": _h,
        "s": _s,
        "sdg": _sdg,
        "t": _t,
        "tdg": _tdg,
        "sx": _sx,
    }
    if key in single_no_param:
        return single_no_param[key]()
    if key in {"rx"}:
        return _rx(_to_float(params[0]))
    if key in {"ry"}:
        return _ry(_to_float(params[0]))
    if key in {"rz"}:
        return _rz(_to_float(params[0]))
    if key in {"p", "phase"}:
        return _phase(_to_float(params[0]))
    if key in {"u", "u3"}:
        return _u3(_to_float(params[0]), _to_float(params[1]), _to_float(params[2]))
    if key in {"swap"}:
        return _swap()
    if key in {"iswap"}:
        return _iswap()
    if key in {"rxx"}:
        return _rxx(_to_float(params[0]))
    if key in {"ryy"}:
        return _ryy(_to_float(params[0]))
    if key in {"rzz"}:
        return _rzz(_to_float(params[0]))
    if key in {"cnot", "cx"}:
        return _controlled(_x(), 1)
    if key == "cy":
        return _controlled(_y(), 1)
    if key == "cz":
        return _controlled(_z(), 1)
    if key == "ch":
        return _controlled(_h(), 1)
    if key in {"cp", "cphase"}:
        return _controlled(_phase(_to_float(params[0])), 1)
    if key == "crx":
        return _controlled(_rx(_to_float(params[0])), 1)
    if key == "cry":
        return _controlled(_ry(_to_float(params[0])), 1)
    if key == "crz":
        return _controlled(_rz(_to_float(params[0])), 1)
    if key == "ccx":
        mat = np.eye(8, dtype=np.complex128)
        mat[6, 6] = 0
        mat[7, 7] = 0
        mat[6, 7] = 1
        mat[7, 6] = 1
        return mat
    if key == "cswap":
        mat = np.eye(8, dtype=np.complex128)
        mat[5, 5] = 0
        mat[6, 6] = 0
        mat[5, 6] = 1
        mat[6, 5] = 1
        return mat
    return None


def zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


def _reorder_operator_for_statevector(operator: np.ndarray, qubits: Sequence[int], n_qubits: int) -> np.ndarray:
    """Embed a k-qubit operator into an n-qubit Hilbert space.

    Qubit 0 is treated as the most significant bit, matching the bitstring format
    used by the rest of this module.  The implementation prioritizes clarity over
    speed; the optional native backends provide fast paths for large circuits.
    """

    qubits = list(qubits)
    k = len(qubits)
    if k == 0:
        return np.eye(2**n_qubits, dtype=np.complex128)
    if operator.shape != (2**k, 2**k):
        raise ValueError(f"Operator shape {operator.shape} does not match {k} target qubits")
    dim = 2**n_qubits
    full = np.zeros((dim, dim), dtype=np.complex128)
    qubit_set = set(qubits)
    spectator_qubits = [q for q in range(n_qubits) if q not in qubit_set]

    for col in range(dim):
        col_bits = [(col >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        local_col = 0
        for q in qubits:
            local_col = (local_col << 1) | col_bits[q]
        spectator_pattern = tuple(col_bits[q] for q in spectator_qubits)
        for local_row in range(2**k):
            amp = operator[local_row, local_col]
            if abs(amp) < _EPS:
                continue
            row_bits = list(col_bits)
            for offset, q in enumerate(reversed(qubits)):
                # local_row is encoded with qubits[0] as most significant bit.
                bit = (local_row >> offset) & 1
                row_bits[q] = bit
            if tuple(row_bits[q] for q in spectator_qubits) != spectator_pattern:
                continue
            row = 0
            for bit in row_bits:
                row = (row << 1) | bit
            full[row, col] += amp
    return full


def apply_operator(state: np.ndarray, operator: np.ndarray, qubits: Sequence[int], n_qubits: int) -> np.ndarray:
    full = _reorder_operator_for_statevector(operator, qubits, n_qubits)
    return full @ state


def _controlled_full_matrix(base: np.ndarray, target_qubits: Sequence[int], control_qubits: Sequence[int], control_values: Sequence[int], n_qubits: int) -> np.ndarray:
    """Create a full-system matrix for a controlled operation."""

    dim = 2**n_qubits
    out = np.eye(dim, dtype=np.complex128)
    target_qubits = list(target_qubits)
    control_qubits = list(control_qubits)
    control_values = list(control_values)
    qubit_set = set(target_qubits)
    for q in control_qubits:
        if q in qubit_set:
            raise ValueError("Control qubits must not overlap target qubits")

    k = len(target_qubits)
    for col in range(dim):
        col_bits = [(col >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if any(col_bits[q] != v for q, v in zip(control_qubits, control_values)):
            continue
        local_col = 0
        for q in target_qubits:
            local_col = (local_col << 1) | col_bits[q]
        # Remove identity rows for this column; controlled block replaces them.
        out[col, col] = 0.0
        for local_row in range(2**k):
            amp = base[local_row, local_col]
            if abs(amp) < _EPS:
                continue
            row_bits = list(col_bits)
            for offset, q in enumerate(reversed(target_qubits)):
                row_bits[q] = (local_row >> offset) & 1
            row = 0
            for bit in row_bits:
                row = (row << 1) | bit
            out[row, col] += amp
    return out


def instruction_matrix(inst, n_qubits: int) -> Tuple[np.ndarray, List[int], bool]:
    """Return ``(matrix, qubits, already_full_system)`` for a HYQ instruction."""

    if getattr(inst, "matrix", None) is not None:
        mat = _complex_array(inst.matrix)
        targets = list(getattr(inst, "qubits", []))
    else:
        mat = gate_matrix(getattr(inst, "name", ""), getattr(inst, "params", []))
        targets = list(getattr(inst, "qubits", []))
    if mat is None:
        raise NotImplementedError(f"Unsupported gate in dense simulator: {getattr(inst, 'name', None)}")

    control_qubits = list(getattr(inst, "control_qubits", []) or [])
    control_values = list(getattr(inst, "control_values", []) or [])
    if control_qubits:
        full = _controlled_full_matrix(mat, targets, control_qubits, control_values, n_qubits)
        return full, list(range(n_qubits)), True
    return mat, targets, False


def simulate_statevector(circuit, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulate a HYQ ``QuantumCircuit`` and return a NumPy state vector."""

    n_qubits = int(circuit.n_qubits)
    state = zero_state(n_qubits) if initial_state is None else _complex_array(initial_state).copy()
    if state.shape != (2**n_qubits,):
        raise ValueError(f"Initial state shape {state.shape} incompatible with {n_qubits} qubits")
    for inst in circuit.instructions:
        if getattr(inst, "circuit", None) is not None:
            # Simulate sub-circuit and apply it as a dense unitary by columns.
            sub = inst.circuit
            sub_dim = 2**sub.n_qubits
            unitary = np.zeros((sub_dim, sub_dim), dtype=np.complex128)
            for j in range(sub_dim):
                basis = np.zeros(sub_dim, dtype=np.complex128)
                basis[j] = 1.0
                unitary[:, j] = simulate_statevector(sub, basis)
            state = apply_operator(state, unitary, list(inst.qubits), n_qubits)
            continue
        name = getattr(inst, "name", "")
        if name == "GlobalPhase":
            theta = _to_float(inst.params[0]) if inst.params else 0.0
            state = np.exp(1.0j * theta) * state
            continue
        mat, qubits, already_full = instruction_matrix(inst, n_qubits)
        if already_full:
            state = mat @ state
        else:
            state = apply_operator(state, mat, qubits, n_qubits)
    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm
    return state


def statevector_to_torch(state: np.ndarray):
    """Convert a NumPy complex vector to the Torch type expected by HYQ backends."""

    if torch is None:
        return state
    return torch.tensor(state, dtype=torch.complex128)


def probability_distribution(state: np.ndarray) -> np.ndarray:
    probs = np.abs(state) ** 2
    total = probs.sum()
    if total <= 0:
        raise ValueError("State vector has zero norm")
    return probs / total


def bitstring(index: int, n_qubits: int, selected: Optional[Sequence[int]] = None) -> str:
    bits = [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
    if selected is None:
        return "".join(str(b) for b in bits)
    return "".join(str(bits[q]) for q in selected)


def sample_counts(state: np.ndarray, shots: int, measure_qubits: Sequence[int], seed: Optional[int] = None) -> Dict[str, int]:
    """Sample computational-basis bitstrings from a state vector."""

    n_qubits = int(round(math.log2(state.size)))
    probs = probability_distribution(state)
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(state.size, size=int(shots), p=probs)
    counts: Dict[str, int] = {}
    for outcome in outcomes:
        key = bitstring(int(outcome), n_qubits, measure_qubits)
        counts[key] = counts.get(key, 0) + 1
    return counts


def expectation_from_pauli_terms(state: np.ndarray, terms: Mapping[Tuple[Tuple[int, str], ...], complex], n_qubits: int) -> complex:
    """Compute <psi|H|psi> for an OpenFermion-like Pauli dictionary."""

    value = 0.0 + 0.0j
    for term, coeff in terms.items():
        if not term:
            value += coeff
            continue
        ops = []
        qubits = []
        for q, pauli in term:
            qubits.append(int(q))
            p = str(pauli).upper()
            if p == "X":
                ops.append(_x())
            elif p == "Y":
                ops.append(_y())
            elif p == "Z":
                ops.append(_z())
            else:
                ops.append(_identity())
        local = ops[0]
        for op in ops[1:]:
            local = np.kron(local, op)
        transformed = apply_operator(state, local, qubits, n_qubits)
        value += coeff * np.vdot(state, transformed)
    return value


@dataclass
class StatevectorResult:
    """Small container used by examples and algorithm modules."""

    state: np.ndarray
    n_qubits: int

    @property
    def probabilities(self) -> np.ndarray:
        return probability_distribution(self.state)

    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None, seed: Optional[int] = None) -> Dict[str, int]:
        return sample_counts(self.state, shots, list(range(self.n_qubits)) if qubits is None else qubits, seed=seed)
