"""Qulacs backend adapter for HYQ-ALG-LIB.

Qulacs is a high-performance C++/Python simulator for variational quantum
circuits.  This adapter provides a native state-vector path when Qulacs is
installed and uses the shared dense simulator for sampling so that the return
format matches the existing backend contract.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import QuantumBackend
from .core import QuantumCircuit
from ._matrix_simulator import gate_matrix, sample_counts, simulate_statevector, statevector_to_torch, _to_float


class QulacsBackend(QuantumBackend):
    """HYQ backend powered by Qulacs.

    The native translator uses Qulacs dense matrices for maximum compatibility
    with the internal HYQ gate set.  Qulacs also has optimized built-in gates;
    dense matrices are chosen here because the project already defines a broad
    collection of gates and control modifiers in one unified format.
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

                gate = DenseMatrix(list(inst.qubits), mat)
                qc.add_gate(gate)
                continue
            if getattr(inst, "name", "") == "GlobalPhase":
                theta = _to_float(inst.params[0]) if inst.params else 0.0
                mat = np.exp(1.0j * theta) * np.eye(1, dtype=np.complex128)
                # Global phase has no direct target; skip because it does not affect
                # measurement and only shifts the global state phase.
                continue
            qc.add_gate(self._dense_gate(inst))
        return qc

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        # Qulacs supports sampling, but the dense utility keeps bit ordering
        # consistent with QiskitBackend and the rest of the library.
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
            return torch.tensor(vector, dtype=torch.complex128)
        except Exception:
            return statevector_to_torch(simulate_statevector(circuit))
