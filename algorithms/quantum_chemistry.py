"""Assorted quantum-chemistry algorithms and analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .classical_eigensolvers import exact_diagonalization, hamiltonian_matrix, hartree_fock_state, normalize_pauli_terms


@dataclass
class MP2Result:
    correlation_energy: float
    total_energy: float
    amplitudes_shape: Tuple[int, int, int, int]


@dataclass
class FidelityResult:
    fidelity: float
    trace_distance_bound: float
    phase: complex


def mp2_energy_from_integrals(eps: np.ndarray, eri_mo: np.ndarray, n_electrons: int, hf_energy: float = 0.0) -> MP2Result:
    """Compute closed-shell MP2 correlation energy from MO energies/integrals.

    The function is intentionally lightweight and serves as a classical reference
    inside the algorithm library.  It assumes spin-orbital-like indexing and uses
    the standard antisymmetrized denominator expression.
    """

    n_orb = len(eps)
    occ = range(n_electrons)
    vir = range(n_electrons, n_orb)
    amplitudes = np.zeros((n_electrons, n_electrons, n_orb - n_electrons, n_orb - n_electrons))
    emp2 = 0.0
    for i in occ:
        for j in occ:
            for a_i, a in enumerate(vir):
                for b_i, b in enumerate(vir):
                    numerator = eri_mo[i, j, a, b] * (2.0 * eri_mo[i, j, a, b] - eri_mo[i, j, b, a])
                    denom = eps[i] + eps[j] - eps[a] - eps[b]
                    if abs(denom) < 1e-12:
                        continue
                    amplitudes[i, j, a_i, b_i] = eri_mo[i, j, a, b] / denom
                    emp2 += numerator / denom
    return MP2Result(float(emp2), float(hf_energy + emp2), amplitudes.shape)


def fci_reference_energy(hamiltonian, n_qubits: Optional[int] = None) -> float:
    """Return the exact lowest eigenvalue of a qubit Hamiltonian."""

    result = exact_diagonalization(hamiltonian, n_qubits=n_qubits, k=1, return_vectors=False)
    return float(result.eigenvalues[0])


def chemical_accuracy_error(estimated_energy: float, reference_energy: float) -> Dict[str, float]:
    hartree_error = float(estimated_energy - reference_energy)
    kcal_per_mol = hartree_error * 627.509474
    return {
        "hartree_error": hartree_error,
        "absolute_hartree_error": abs(hartree_error),
        "kcal_per_mol_error": kcal_per_mol,
        "within_chemical_accuracy": abs(kcal_per_mol) <= 1.0,
    }


def state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> FidelityResult:
    a = np.asarray(state_a, dtype=np.complex128)
    b = np.asarray(state_b, dtype=np.complex128)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    amp = np.vdot(a, b)
    fid = float(abs(amp) ** 2)
    return FidelityResult(fid, float(np.sqrt(max(0.0, 1.0 - fid))), amp)


def natural_orbital_occupations(one_rdm: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(np.asarray(one_rdm, dtype=np.complex128))
    return np.sort(vals.real)[::-1]


def particle_number_expectation(state: np.ndarray, n_qubits: int) -> float:
    probs = np.abs(state) ** 2
    exp = 0.0
    for idx, p in enumerate(probs):
        exp += p * bin(idx).count("1")
    return float(exp)


def spin_z_expectation(state: np.ndarray, n_qubits: int) -> float:
    probs = np.abs(state) ** 2
    exp = 0.0
    for idx, p in enumerate(probs):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        alpha = sum(bits[0::2])
        beta = sum(bits[1::2])
        exp += p * 0.5 * (alpha - beta)
    return float(exp)


def active_space_indices(n_orbitals: int, n_core: int = 0, n_active: Optional[int] = None, n_virtual: int = 0) -> List[int]:
    if n_active is None:
        stop = n_orbitals - n_virtual
    else:
        stop = n_core + n_active
    return list(range(n_core, min(stop, n_orbitals)))


def freeze_core_energy_shift(core_orbital_energies: Sequence[float]) -> float:
    return float(2.0 * np.sum(np.asarray(core_orbital_energies, dtype=float)))


def symmetry_project_state(state: np.ndarray, n_qubits: int, n_particles: Optional[int] = None, sz: Optional[float] = None) -> np.ndarray:
    out = np.zeros_like(state, dtype=np.complex128)
    for idx, amp in enumerate(state):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if n_particles is not None and sum(bits) != n_particles:
            continue
        if sz is not None:
            alpha = sum(bits[0::2])
            beta = sum(bits[1::2])
            if abs(0.5 * (alpha - beta) - sz) > 1e-12:
                continue
        out[idx] = amp
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def commutator_norm(op_a: np.ndarray, op_b: np.ndarray) -> float:
    c = op_a @ op_b - op_b @ op_a
    return float(np.linalg.norm(c))
