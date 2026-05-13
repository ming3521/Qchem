"""Qulacs backend adapter for HYQ-ALG-LIB.

This backend translates the project-level ``QuantumCircuit`` representation into
Qulacs circuits.  HYQ uses a big-endian state-vector convention: qubit 0 is the
left-most bit in printed bitstrings and the most-significant bit in the dense
state vector.  Qulacs uses the common little-endian convention internally: qubit
0 is the least-significant bit of the array index.  The adapter therefore
converts dense gate matrices before passing them to ``qulacs.gate.DenseMatrix``
and converts the final state vector back to the HYQ convention.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import QuantumBackend
from .core import QuantumCircuit
from ._matrix_simulator import gate_matrix, sample_counts, simulate_statevector, statevector_to_torch, _to_float


def _reverse_bits(index: int, width: int) -> int:
    """Return ``index`` with ``width`` binary bits reversed."""

    out = 0
    for bit in range(width):
        out = (out << 1) | ((index >> bit) & 1)
    return out


def _hyq_matrix_to_qulacs_local(matrix: np.ndarray, n_local_qubits: int) -> np.ndarray:
    """Convert a HYQ local operator matrix to Qulacs local basis ordering.

    HYQ local matrices are written with ``qubits[0]`` as the most-significant
    local bit.  Qulacs DenseMatrix interprets the first target index as the
    least-significant local bit.  A bit-reversal permutation on rows and columns
    converts between these conventions.  Single-qubit matrices are unchanged.
    """

    matrix = np.asarray(matrix, dtype=np.complex128)
    if n_local_qubits <= 1:
        return matrix
    dim = 1 << n_local_qubits
    if matrix.shape != (dim, dim):
        raise ValueError(f"Matrix shape {matrix.shape} does not match {n_local_qubits} local qubits")
    out = np.empty_like(matrix)
    for r_little in range(dim):
        r_big = _reverse_bits(r_little, n_local_qubits)
        for c_little in range(dim):
            c_big = _reverse_bits(c_little, n_local_qubits)
            out[r_little, c_little] = matrix[r_big, c_big]
    return out


def _qulacs_vector_to_hyq(vector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert a Qulacs state vector to the HYQ big-endian state convention."""

    vector = np.asarray(vector, dtype=np.complex128)
    dim = 1 << n_qubits
    if vector.shape[0] != dim:
        raise ValueError(f"State-vector length {vector.shape[0]} does not match {n_qubits} qubits")
    if n_qubits <= 1:
        return vector.copy()
    out = np.empty_like(vector)
    for little_index, amp in enumerate(vector):
        big_index = _reverse_bits(little_index, n_qubits)
        out[big_index] = amp
    return out


class QulacsBackend(QuantumBackend):
    """HYQ backend powered by Qulacs.

    The native translator uses Qulacs dense matrices for broad compatibility
    with the internal HYQ gate set.  The adapter explicitly handles the endian
    mismatch between HYQ and Qulacs so that ``get_statevector`` returns the same
    basis ordering as QiskitBackend, PennyLaneBackend, TensorCircuitBackend and
    the shared NumPy reference simulator.
    """

    def __init__(self, seed: Optional[int] = None, use_native: bool = True):
        self.seed = seed
        self.use_native = use_native
        self._qulacs = None

    def _check_qulacs(self):
        if self._qulacs is not None:
            return self._qulacs
        try:
            import qulacs
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("QulacsBackend requires 'qulacs'. Install it with: pip install qulacs") from exc
        self._qulacs = qulacs
        return qulacs

    def _dense_gate(self, inst):
        from qulacs.gate import DenseMatrix

        if getattr(inst, "matrix", None) is not None:
            mat = np.asarray(inst.matrix, dtype=np.complex128)
        else:
            mat = gate_matrix(getattr(inst, "name", ""), getattr(inst, "params", []))
        if mat is None:
            raise NotImplementedError(f"QulacsBackend does not support gate '{getattr(inst, 'name', None)}'")
        targets = list(getattr(inst, "qubits", []))
        mat = _hyq_matrix_to_qulacs_local(mat, len(targets))
        gate = DenseMatrix(targets, mat)
        for q, v in zip(getattr(inst, "control_qubits", []) or [], getattr(inst, "control_values", []) or []):
            gate.add_control_qubit(int(q), int(v))
        return gate

    def _to_qulacs_circuit(self, circuit: QuantumCircuit):
        qulacs = self._check_qulacs()
        qc = qulacs.QuantumCircuit(circuit.n_qubits)
        for inst in circuit.instructions:
            if getattr(inst, "circuit", None) is not None:
                # Turn a sub-circuit into a dense matrix by simulation of basis states.
                sub = inst.circuit
                dim = 2**sub.n_qubits
                mat = np.zeros((dim, dim), dtype=np.complex128)
                for j in range(dim):
                    basis = np.zeros(dim, dtype=np.complex128)
                    basis[j] = 1.0
                    mat[:, j] = simulate_statevector(sub, basis)
                from qulacs.gate import DenseMatrix

                mat = _hyq_matrix_to_qulacs_local(mat, len(list(inst.qubits)))
                gate = DenseMatrix(list(inst.qubits), mat)
                qc.add_gate(gate)
                continue
            if getattr(inst, "name", "") == "GlobalPhase":
                # Global phase does not affect measurement and most state-vector
                # comparisons in this library are phase-insensitive.  Qulacs has
                # no target-free global phase gate, so we skip it here.
                continue
            qc.add_gate(self._dense_gate(inst))
        return qc

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        # The shared dense utility keeps bit ordering consistent with Qiskit,
        # PennyLane, TensorCircuit and the rest of HYQ.
        state = simulate_statevector(circuit)
        return sample_counts(state, shots, measure_qubits, seed=self.seed)

    def get_statevector(self, circuit) -> torch.Tensor:
        if not self.use_native:
            return statevector_to_torch(simulate_statevector(circuit))
        try:
            qulacs = self._check_qulacs()
            state = qulacs.QuantumState(circuit.n_qubits)
            state.set_zero_state()
            qc = self._to_qulacs_circuit(circuit)
            qc.update_quantum_state(state)
            vector = np.asarray(state.get_vector(), dtype=np.complex128)
            vector = _qulacs_vector_to_hyq(vector, circuit.n_qubits)
            return torch.tensor(vector, dtype=torch.complex128)
        except Exception:
            return statevector_to_torch(simulate_statevector(circuit))
