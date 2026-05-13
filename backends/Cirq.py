"""Cirq backend adapter for HYQ-ALG-LIB.

This backend adds a fourth execution target besides Qiskit, PennyLane and
TensorCircuit.  It follows the same ``QuantumBackend`` interface used by the
original project: ``run_sampling`` returns measurement counts and
``get_statevector`` returns a Torch complex tensor.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .base import QuantumBackend
from .core import QuantumCircuit
from ._matrix_simulator import simulate_statevector, statevector_to_torch, _to_float


class CirqBackend(QuantumBackend):
    """Backend implemented with Google's Cirq simulator.

    Cirq is useful when one wants fine control over device topology and moment
    structure.  This adapter translates the HYQ circuit into Cirq operations when
    possible.  If a user appends a custom dense unitary or a gate not directly
    available in Cirq, the state-vector mode falls back to the shared dense
    simulator so that the result remains defined.
    """

    def __init__(self, dtype=np.complex128, seed: Optional[int] = None):
        self.dtype = dtype
        self.seed = seed
        self._cirq = None
        self._simulator = None

    def _check_cirq(self):
        if self._cirq is not None:
            return self._cirq
        try:
            import cirq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("CirqBackend requires 'cirq'. Install it with: pip install cirq") from exc
        self._cirq = cirq
        self._simulator = cirq.Simulator(dtype=self.dtype, seed=self.seed)
        return cirq

    @staticmethod
    def _as_angle(params: Sequence, index: int = 0) -> float:
        if not params or index >= len(params):
            return 0.0
        return _to_float(params[index])

    def _qubits(self, n_qubits: int):
        cirq = self._check_cirq()
        return [cirq.LineQubit(i) for i in range(n_qubits)]

    def _append_controlled(self, op, inst, qreg):
        cirq = self._check_cirq()
        controls = list(getattr(inst, "control_qubits", []) or [])
        values = list(getattr(inst, "control_values", []) or [])
        if not controls:
            return op
        if values and any(v not in (0, 1) for v in values):
            raise ValueError("CirqBackend only supports binary control values")
        control_qs = [qreg[q] for q in controls]
        if not values or all(v == 1 for v in values):
            return op.controlled_by(*control_qs)
        # For zero-controls, conjugate the corresponding controls by X gates.
        before = []
        after = []
        for q, v in zip(control_qs, values):
            if v == 0:
                before.append(cirq.X(q))
                after.append(cirq.X(q))
        return before + [op.controlled_by(*control_qs)] + after

    def _instruction_to_ops(self, inst, qreg):
        cirq = self._check_cirq()
        name = (getattr(inst, "name", "") or "").lower()
        params = list(getattr(inst, "params", []) or [])
        qs = [qreg[i] for i in getattr(inst, "qubits", [])]

        if getattr(inst, "matrix", None) is not None:
            gate = cirq.MatrixGate(np.asarray(inst.matrix, dtype=np.complex128))
            return [self._append_controlled(gate.on(*qs), inst, qreg)]
        if getattr(inst, "circuit", None) is not None:
            sub = self._to_cirq_circuit(inst.circuit)
            gate = cirq.CircuitOperation(sub.freeze())
            return [gate.with_qubits(*qs)]
        if name == "globalphase":
            return [cirq.GlobalPhaseOperation(np.exp(1.0j * self._as_angle(params)))]

        op = None
        if name in {"id", "i"}:
            op = cirq.I(qs[0])
        elif name == "x":
            op = cirq.X(qs[0])
        elif name == "y":
            op = cirq.Y(qs[0])
        elif name == "z":
            op = cirq.Z(qs[0])
        elif name == "h":
            op = cirq.H(qs[0])
        elif name == "s":
            op = cirq.S(qs[0])
        elif name == "sdg":
            op = cirq.S(qs[0]) ** -1
        elif name == "t":
            op = cirq.T(qs[0])
        elif name == "tdg":
            op = cirq.T(qs[0]) ** -1
        elif name == "rx":
            op = cirq.rx(self._as_angle(params))(qs[0])
        elif name == "ry":
            op = cirq.ry(self._as_angle(params))(qs[0])
        elif name == "rz":
            op = cirq.rz(self._as_angle(params))(qs[0])
        elif name in {"p", "phase"}:
            op = cirq.ZPowGate(exponent=self._as_angle(params) / np.pi).on(qs[0])
        elif name == "u3":
            theta, phi, lam = [self._as_angle(params, i) for i in range(3)]
            op = cirq.MatrixGate(
                np.array(
                    [
                        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
                    ],
                    dtype=np.complex128,
                )
            ).on(qs[0])
        elif name in {"cx", "cnot"}:
            op = cirq.CNOT(qs[0], qs[1])
        elif name == "cy":
            op = cirq.ControlledGate(cirq.Y).on(qs[0], qs[1])
        elif name == "cz":
            op = cirq.CZ(qs[0], qs[1])
        elif name == "swap":
            op = cirq.SWAP(qs[0], qs[1])
        elif name == "iswap":
            op = cirq.ISWAP(qs[0], qs[1])
        elif name in {"cp", "cphase"}:
            op = cirq.ControlledGate(cirq.ZPowGate(exponent=self._as_angle(params) / np.pi)).on(qs[0], qs[1])
        elif name == "crx":
            op = cirq.ControlledGate(cirq.rx(self._as_angle(params))).on(qs[0], qs[1])
        elif name == "cry":
            op = cirq.ControlledGate(cirq.ry(self._as_angle(params))).on(qs[0], qs[1])
        elif name == "crz":
            op = cirq.ControlledGate(cirq.rz(self._as_angle(params))).on(qs[0], qs[1])
        elif name == "rxx":
            op = cirq.XXPowGate(exponent=self._as_angle(params) / np.pi).on(qs[0], qs[1])
        elif name == "ryy":
            op = cirq.YYPowGate(exponent=self._as_angle(params) / np.pi).on(qs[0], qs[1])
        elif name == "rzz":
            op = cirq.ZZPowGate(exponent=self._as_angle(params) / np.pi).on(qs[0], qs[1])
        elif name == "ccx":
            op = cirq.TOFFOLI(qs[0], qs[1], qs[2])
        elif name == "cswap":
            op = cirq.CSWAP(qs[0], qs[1], qs[2])
        else:
            raise NotImplementedError(f"CirqBackend does not support gate '{inst.name}' natively")

        controlled = self._append_controlled(op, inst, qreg)
        return controlled if isinstance(controlled, list) else [controlled]

    def _to_cirq_circuit(self, circuit: QuantumCircuit):
        cirq = self._check_cirq()
        qreg = self._qubits(circuit.n_qubits)
        ops = []
        for inst in circuit.instructions:
            ops.extend(self._instruction_to_ops(inst, qreg))
        return cirq.Circuit(ops)

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        cirq = self._check_cirq()
        qreg = self._qubits(circuit.n_qubits)
        cirq_circuit = self._to_cirq_circuit(circuit)
        for q in measure_qubits:
            cirq_circuit.append(cirq.measure(qreg[q], key=f"q{q}"))
        result = self._simulator.run(cirq_circuit, repetitions=shots)
        counts: Dict[str, int] = {}
        for row in range(shots):
            bits = []
            for q in measure_qubits:
                bits.append(str(int(result.measurements[f"q{q}"][row, 0])))
            key = "".join(bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_statevector(self, circuit) -> torch.Tensor:
        try:
            cirq_circuit = self._to_cirq_circuit(circuit)
            qubits = self._qubits(circuit.n_qubits)
            result = self._simulator.simulate(cirq_circuit, qubit_order=qubits)
            return torch.tensor(np.asarray(result.final_state_vector, dtype=np.complex128), dtype=torch.complex128)
        except Exception:
            # Preserve a functional state-vector mode for custom MatrixGate or gates
            # that are present in HYQ but absent from a user's installed Cirq version.
            return statevector_to_torch(simulate_statevector(circuit))
