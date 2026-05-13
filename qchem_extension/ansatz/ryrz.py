"""RyRz universal ansatz for HYQ-ALG-LIB.

The software manual lists RyRz as a planned high-expressibility ansatz.  This
implementation follows the project's existing ``Ansatz`` API and emits the same
backend-independent ``QuantumCircuit`` object as HEA and UCCSD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from ansatz.base import Ansatz
from backends.core import QuantumCircuit


Entangler = Tuple[int, int]


def _as_tensor(params):
    if isinstance(params, torch.Tensor):
        return params
    return torch.tensor(params, dtype=torch.float64)


def _validate_entanglement(name: str):
    valid = {"linear", "reverse_linear", "circular", "full", "alternating", "none", "rzz"}
    if name not in valid:
        raise ValueError(f"Unknown entanglement '{name}'. Valid values are {sorted(valid)}")


def entangler_pairs(n_qubits: int, entanglement: str, layer: int = 0) -> List[Entangler]:
    """Return control-target pairs for a layer.

    ``alternating`` uses even bonds in even layers and odd bonds in odd layers,
    which is friendly to nearest-neighbor hardware because gates within a layer do
    not overlap.
    """

    _validate_entanglement(entanglement)
    if n_qubits < 2 or entanglement == "none":
        return []
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entanglement == "reverse_linear":
        return [(i + 1, i) for i in range(n_qubits - 1)]
    if entanglement == "circular":
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if entanglement == "alternating":
        start = layer % 2
        return [(i, i + 1) for i in range(start, n_qubits - 1, 2)]
    if entanglement == "rzz":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    return []


@dataclass
class RyRzLayerSpec:
    """Describes a RyRz layer used for reproducible circuit generation."""

    layer: int
    rotation_start: int
    rotation_stop: int
    entangler_pairs: List[Entangler]
    entangler_param_start: Optional[int] = None
    entangler_param_stop: Optional[int] = None


class RyRzAnsatz(Ansatz):
    """Layered Rz-Ry-Rz ansatz with configurable entanglement.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the circuit.
    n_layers:
        Number of repeated RyRz blocks.
    entanglement:
        One of ``linear``, ``reverse_linear``, ``circular``, ``full``,
        ``alternating``, ``none`` or ``rzz``.  ``rzz`` uses parameterized RZZ
        entanglers and therefore adds one trainable parameter per bond per layer.
    initial_state:
        ``zero`` leaves the all-zero state unchanged.  ``hf`` prepares a
        Hartree-Fock occupation pattern using ``n_electrons``.
    final_layer:
        Whether to append a final Rz-Ry-Rz rotation layer after the last
        entangler block.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        entanglement: str = "linear",
        initial_state: str = "zero",
        n_electrons: Optional[int] = None,
        final_layer: bool = True,
    ):
        _validate_entanglement(entanglement)
        self.n_layers = int(n_layers)
        self.entanglement = entanglement
        self.initial_state = initial_state
        self.n_electrons = n_electrons
        self.final_layer = bool(final_layer)
        self.layer_specs: List[RyRzLayerSpec] = []

        n_rot_layers = self.n_layers + (1 if self.final_layer else 0)
        n_rotation_params = 3 * n_qubits * n_rot_layers
        n_entangler_params = 0
        if entanglement == "rzz":
            n_entangler_params = sum(len(entangler_pairs(n_qubits, entanglement, l)) for l in range(self.n_layers))
        super().__init__(n_qubits, n_rotation_params + n_entangler_params)
        self._build_layer_specs()

    def _build_layer_specs(self):
        cursor = 0
        ent_cursor = 3 * self.n_qubits * (self.n_layers + (1 if self.final_layer else 0))
        for layer in range(self.n_layers):
            start = cursor
            stop = start + 3 * self.n_qubits
            cursor = stop
            pairs = entangler_pairs(self.n_qubits, self.entanglement, layer)
            ent_start = ent_stop = None
            if self.entanglement == "rzz":
                ent_start = ent_cursor
                ent_stop = ent_start + len(pairs)
                ent_cursor = ent_stop
            self.layer_specs.append(RyRzLayerSpec(layer, start, stop, pairs, ent_start, ent_stop))
        if self.final_layer:
            self.final_rotation_slice = slice(cursor, cursor + 3 * self.n_qubits)
        else:
            self.final_rotation_slice = slice(cursor, cursor)

    def _prepare_initial_state(self, qc: QuantumCircuit):
        state = self.initial_state.lower()
        if state in {"zero", "zeros", "0"}:
            return
        if state in {"hf", "hartree_fock", "hartree-fock"}:
            if self.n_electrons is None:
                raise ValueError("n_electrons must be provided for Hartree-Fock initial state")
            for q in range(min(self.n_electrons, self.n_qubits)):
                qc.x(q)
            return
        if state.startswith("bitstring:"):
            bitstr = state.split(":", 1)[1]
            if len(bitstr) != self.n_qubits:
                raise ValueError("bitstring initial_state length must equal n_qubits")
            for q, bit in enumerate(bitstr):
                if bit == "1":
                    qc.x(q)
            return
        raise ValueError(f"Unsupported initial_state '{self.initial_state}'")

    def _apply_rotation_layer(self, qc: QuantumCircuit, params: torch.Tensor, start: int):
        local = params[start : start + 3 * self.n_qubits].view(self.n_qubits, 3)
        for q in range(self.n_qubits):
            qc.rz(q, local[q, 0])
            qc.ry(q, local[q, 1])
            qc.rz(q, local[q, 2])

    def _apply_entangler_layer(self, qc: QuantumCircuit, params: torch.Tensor, spec: RyRzLayerSpec):
        if self.entanglement == "rzz":
            assert spec.entangler_param_start is not None
            ent_params = params[spec.entangler_param_start : spec.entangler_param_stop]
            for theta, (a, b) in zip(ent_params, spec.entangler_pairs):
                qc.rzz(a, b, theta)
            return
        for a, b in spec.entangler_pairs:
            qc.cx(a, b)

    def forward(self, params) -> QuantumCircuit:
        params = _as_tensor(params)
        if params.numel() != self.n_params:
            raise ValueError(f"RyRzAnsatz expected {self.n_params} parameters, got {params.numel()}")
        qc = QuantumCircuit(self.n_qubits, name="RyRzAnsatz")
        self._prepare_initial_state(qc)
        for spec in self.layer_specs:
            self._apply_rotation_layer(qc, params, spec.rotation_start)
            self._apply_entangler_layer(qc, params, spec)
        if self.final_layer:
            self._apply_rotation_layer(qc, params, self.final_rotation_slice.start)
        return qc

    def parameter_shape(self) -> Tuple[int]:
        return (self.n_params,)

    def zero_parameters(self) -> torch.Tensor:
        return torch.zeros(self.n_params, dtype=torch.float64)

    def random_parameters(self, scale: float = 0.01, seed: Optional[int] = None) -> torch.Tensor:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        return scale * torch.randn(self.n_params, dtype=torch.float64, generator=generator)
