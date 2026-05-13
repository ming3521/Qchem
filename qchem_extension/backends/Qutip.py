"""QuTiP / qutip-qip backend adapter for HYQ-ALG-LIB.

QuTiP is particularly useful for open-system and physics-layer simulations.  This
adapter keeps the public HYQ backend API simple while making it possible to use a
QuTiP-based environment for exact state-vector checks and sampling experiments.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import QuantumBackend
from ._matrix_simulator import sample_counts, simulate_statevector, statevector_to_torch


class QutipBackend(QuantumBackend):
    """Backend that validates the QuTiP dependency and uses dense state vectors.

    The current HYQ circuit abstraction is closer to a gate list than to a QuTiP
    ``QubitCircuit`` object.  Instead of forcing a lossy conversion, the adapter
    relies on the shared dense simulator for deterministic state-vector results.
    If QuTiP is installed, users can wrap the returned vector as ``qutip.Qobj`` via
    ``to_qobj`` for further open-system analysis.
    """

    def __init__(self, seed: Optional[int] = None, require_qutip: bool = True):
        self.seed = seed
        self.require_qutip = require_qutip
        self._qutip = None

    def _check_qutip(self):
        if self._qutip is not None:
            return self._qutip
        try:
            import qutip
        except ImportError as exc:  # pragma: no cover - optional dependency
            if self.require_qutip:
                raise ImportError("QutipBackend requires 'qutip'. Install it with: pip install qutip qutip-qip") from exc
            return None
        self._qutip = qutip
        return qutip

    def to_qobj(self, circuit):
        """Return the final state as a QuTiP ket object."""

        qutip = self._check_qutip()
        if qutip is None:
            raise ImportError("qutip is not installed")
        state = simulate_statevector(circuit)
        return qutip.Qobj(state.reshape((-1, 1)), dims=[[2] * circuit.n_qubits, [1] * circuit.n_qubits])

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        self._check_qutip()
        state = simulate_statevector(circuit)
        return sample_counts(state, shots, measure_qubits, seed=self.seed)

    def get_statevector(self, circuit) -> torch.Tensor:
        self._check_qutip()
        return statevector_to_torch(simulate_statevector(circuit))
