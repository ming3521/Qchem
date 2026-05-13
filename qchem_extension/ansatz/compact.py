"""Compact chemistry-inspired ansatz families.

The manual describes a compact ansatz based on electron-pair excitations and
k-UPCCG-style parameter reduction.  The classes below provide practical circuit
builders that keep the API consistent with ``Ansatz`` while adding enough
metadata for VQE, VarQITE and diagnostic tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from ansatz.base import Ansatz
from backends.core import QuantumCircuit


Pair = Tuple[int, int]
PairMove = Tuple[int, int]


def _as_tensor(params):
    if isinstance(params, torch.Tensor):
        return params
    return torch.tensor(params, dtype=torch.float64)


@dataclass(frozen=True)
class PairExcitation:
    source_orbital: int
    target_orbital: int
    alpha_source: int
    beta_source: int
    alpha_target: int
    beta_target: int

    def qubits(self) -> Tuple[int, int, int, int]:
        return (self.alpha_source, self.beta_source, self.alpha_target, self.beta_target)


class PairExcitationAnsatz(Ansatz):
    """Pair-excitation ansatz for closed-shell chemistry.

    A spatial orbital ``p`` is mapped to spin orbitals ``2p`` and ``2p+1``.
    Each parameter approximately transfers an alpha-beta electron pair from an
    occupied pair orbital into a virtual pair orbital.  The circuit block is a
    shallow number-conserving pattern composed of CNOT, Ry and RZ/RZZ rotations.
    It is intentionally compact: parameter count scales as occupied_pairs times
    virtual_pairs, rather than the full UCCSD singles+doubles pool.
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        active_pairs: Optional[Sequence[PairMove]] = None,
        repetitions: int = 1,
        include_pair_phase: bool = True,
    ):
        if n_qubits % 2 != 0:
            raise ValueError("PairExcitationAnsatz expects an even number of spin-orbital qubits")
        if n_electrons % 2 != 0:
            raise ValueError("PairExcitationAnsatz is intended for closed-shell even-electron systems")
        self.n_electrons = int(n_electrons)
        self.n_spatial_orbitals = n_qubits // 2
        self.n_occupied_pairs = n_electrons // 2
        self.repetitions = int(repetitions)
        self.include_pair_phase = bool(include_pair_phase)
        self.excitations = self._make_excitations(active_pairs)
        params_per_excitation = 2 if self.include_pair_phase else 1
        super().__init__(n_qubits, len(self.excitations) * params_per_excitation * self.repetitions)

    def _make_excitations(self, active_pairs: Optional[Sequence[PairMove]]) -> List[PairExcitation]:
        if active_pairs is None:
            moves = [
                (i, a)
                for i in range(self.n_occupied_pairs)
                for a in range(self.n_occupied_pairs, self.n_spatial_orbitals)
            ]
        else:
            moves = list(active_pairs)
        excitations: List[PairExcitation] = []
        for source, target in moves:
            if source == target:
                continue
            if source < 0 or target < 0 or source >= self.n_spatial_orbitals or target >= self.n_spatial_orbitals:
                raise ValueError(f"Invalid pair excitation ({source}, {target})")
            excitations.append(
                PairExcitation(
                    source,
                    target,
                    2 * source,
                    2 * source + 1,
                    2 * target,
                    2 * target + 1,
                )
            )
        return excitations

    def _prepare_hf(self, qc: QuantumCircuit):
        for q in range(min(self.n_electrons, self.n_qubits)):
            qc.x(q)

    def _append_pair_block(self, qc: QuantumCircuit, excitation: PairExcitation, theta, phase=None):
        p_a, p_b, q_a, q_b = excitation.qubits()
        # Alpha channel Givens-style mixing.
        qc.cx(p_a, q_a)
        qc.ry(q_a, theta)
        qc.cx(p_a, q_a)
        # Beta channel with the same amplitude to preserve closed-shell pairing.
        qc.cx(p_b, q_b)
        qc.ry(q_b, theta)
        qc.cx(p_b, q_b)
        # Correlate the two spin channels.  RZZ is native in the internal circuit
        # and can be compiled by Qiskit or executed by the new dense backends.
        qc.rzz(q_a, q_b, theta)
        qc.rzz(p_a, p_b, -theta)
        if phase is not None:
            qc.rz(q_a, phase)
            qc.rz(q_b, phase)
            qc.rz(p_a, -phase)
            qc.rz(p_b, -phase)

    def forward(self, params) -> QuantumCircuit:
        params = _as_tensor(params)
        if params.numel() != self.n_params:
            raise ValueError(f"PairExcitationAnsatz expected {self.n_params} parameters, got {params.numel()}")
        qc = QuantumCircuit(self.n_qubits, name="PairExcitationAnsatz")
        self._prepare_hf(qc)
        cursor = 0
        for _ in range(self.repetitions):
            for excitation in self.excitations:
                theta = params[cursor]
                cursor += 1
                phase = None
                if self.include_pair_phase:
                    phase = params[cursor]
                    cursor += 1
                self._append_pair_block(qc, excitation, theta, phase)
        return qc

    @property
    def n_pair_excitations(self) -> int:
        return len(self.excitations)

    def zero_parameters(self) -> torch.Tensor:
        return torch.zeros(self.n_params, dtype=torch.float64)


class KUpCCGAnsatz(Ansatz):
    """k-UpCCG inspired ansatz.

    This class generalizes pair excitations by repeating a compact generalized
    pair cluster layer ``k`` times.  It includes all pair transfers between
    spatial orbitals instead of only occupied-to-virtual transfers, which is a
    common practical k-UpCCG variant.
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        k: int = 1,
        include_orbital_rotations: bool = True,
    ):
        if n_qubits % 2 != 0:
            raise ValueError("KUpCCGAnsatz expects spin-orbital qubits in alpha/beta pairs")
        self.n_electrons = int(n_electrons)
        self.k = int(k)
        self.include_orbital_rotations = bool(include_orbital_rotations)
        self.n_spatial_orbitals = n_qubits // 2
        self.pair_moves = [(p, q) for p in range(self.n_spatial_orbitals) for q in range(p + 1, self.n_spatial_orbitals)]
        pair_params = len(self.pair_moves)
        orbital_params = self.n_spatial_orbitals * (self.n_spatial_orbitals - 1) // 2 if include_orbital_rotations else 0
        super().__init__(n_qubits, self.k * (pair_params + orbital_params))

    def _prepare_hf(self, qc: QuantumCircuit):
        for q in range(min(self.n_electrons, self.n_qubits)):
            qc.x(q)

    def _append_pair_transfer(self, qc: QuantumCircuit, p: int, q: int, theta):
        p_a, p_b, q_a, q_b = 2 * p, 2 * p + 1, 2 * q, 2 * q + 1
        qc.rxx(p_a, q_a, theta)
        qc.ryy(p_a, q_a, theta)
        qc.rxx(p_b, q_b, theta)
        qc.ryy(p_b, q_b, theta)
        qc.rzz(p_a, p_b, theta / 2)
        qc.rzz(q_a, q_b, theta / 2)

    def _append_orbital_rotation(self, qc: QuantumCircuit, p: int, q: int, theta):
        # Spin-adapted orbital rotation applied to alpha and beta channels.
        qc.rxx(2 * p, 2 * q, theta)
        qc.ryy(2 * p, 2 * q, theta)
        qc.rxx(2 * p + 1, 2 * q + 1, theta)
        qc.ryy(2 * p + 1, 2 * q + 1, theta)

    def forward(self, params) -> QuantumCircuit:
        params = _as_tensor(params)
        if params.numel() != self.n_params:
            raise ValueError(f"KUpCCGAnsatz expected {self.n_params} parameters, got {params.numel()}")
        qc = QuantumCircuit(self.n_qubits, name="KUpCCGAnsatz")
        self._prepare_hf(qc)
        cursor = 0
        for _ in range(self.k):
            for p, q in self.pair_moves:
                self._append_pair_transfer(qc, p, q, params[cursor])
                cursor += 1
            if self.include_orbital_rotations:
                for p, q in self.pair_moves:
                    self._append_orbital_rotation(qc, p, q, params[cursor])
                    cursor += 1
        return qc

    def zero_parameters(self) -> torch.Tensor:
        return torch.zeros(self.n_params, dtype=torch.float64)


# Backward-friendly aliases for the manual's naming.
CompactAnsatz = PairExcitationAnsatz
kUPCCGAnsatz = KUpCCGAnsatz
