"""Assorted quantum-chemistry algorithms and analysis helpers.

This module intentionally keeps the helpers lightweight.  They are mainly useful
as classical references and diagnostics for variational / QITE-style algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .classical_eigensolvers import exact_diagonalization


@dataclass
class MP2Result:
    correlation_energy: float
    total_energy: float
    amplitudes_shape: Tuple[int, int, int, int]
    skipped_denominators: int = 0
    convention: str = "closed_shell_chemist"


@dataclass
class FidelityResult:
    fidelity: float
    trace_distance_bound: float
    phase: complex


def _validate_n_qubits_state(state: np.ndarray, n_qubits: Optional[int] = None) -> Tuple[np.ndarray, int]:
    vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vec.size == 0:
        raise ValueError("state must not be empty")
    log_dim = np.log2(vec.size)
    inferred = int(round(log_dim))
    if 2**inferred != vec.size:
        raise ValueError(f"state length {vec.size} is not a power of two")
    if n_qubits is None:
        n_qubits = inferred
    elif int(n_qubits) != inferred:
        raise ValueError(f"state length {vec.size} is incompatible with n_qubits={n_qubits}")
    norm = np.linalg.norm(vec)
    if norm <= 0:
        raise ValueError("state must have nonzero norm")
    return vec / norm, int(n_qubits)


def _validate_integrals(eps: np.ndarray, eri_mo: np.ndarray, n_electrons: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    eps_arr = np.asarray(eps, dtype=float).reshape(-1)
    eri_arr = np.asarray(eri_mo)
    n_orb = eps_arr.size
    if eri_arr.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError(
            f"eri_mo shape {eri_arr.shape} is incompatible with len(eps)={n_orb}; "
            "expected (n_orb, n_orb, n_orb, n_orb)"
        )
    n_e = int(n_electrons)
    if n_e < 0 or n_e > n_orb:
        raise ValueError(f"n_electrons={n_electrons} must be in [0, {n_orb}]")
    return eps_arr, eri_arr, n_orb, n_e


def mp2_energy_from_integrals(
    eps: np.ndarray,
    eri_mo: np.ndarray,
    n_electrons: int,
    hf_energy: float = 0.0,
    *,
    integral_order: str = "chemist",
    spin_orbital: bool = False,
    denominator_tol: float = 1e-12,
) -> MP2Result:
    """Compute an MP2 reference energy from orbital energies and MO integrals.

    Parameters
    ----------
    eps:
        Orbital energies.
    eri_mo:
        Four-index MO electron-repulsion integrals.
    n_electrons:
        Number of occupied orbitals in the indexing convention being used.  For
        the default closed-shell spatial formula this is the number of occupied
        spatial orbitals.  For ``spin_orbital=True`` this is the number of
        occupied spin orbitals.
    hf_energy:
        Hartree-Fock reference energy to which the MP2 correlation correction is
        added.
    integral_order:
        ``"chemist"`` means common spatial-orbital integrals ``eri[p,q,r,s] =
        (pq|rs)`` and uses the closed-shell formula with ``(ia|jb)`` and
        ``(ib|ja)``.  ``"legacy_ijab"`` preserves the indexing used by the
        original version, where the direct term is read as ``eri[i,j,a,b]``.
    spin_orbital:
        If true, use the spin-orbital MP2 expression with antisymmetrized matrix
        elements built as ``eri[i,j,a,b] - eri[i,j,b,a]``.  In this mode
        ``integral_order`` must be ``"legacy_ijab"`` or ``"spin_ijab"``.
    denominator_tol:
        Denominators with absolute value below this threshold are skipped.
    """

    eps_arr, eri_arr, n_orb, n_occ = _validate_integrals(eps, eri_mo, n_electrons)
    n_vir = n_orb - n_occ
    amplitudes = np.zeros((n_occ, n_occ, n_vir, n_vir), dtype=np.result_type(eri_arr, float))
    emp2 = 0.0 + 0.0j
    skipped = 0

    order = integral_order.lower()
    occ = range(n_occ)
    vir = range(n_occ, n_orb)

    if spin_orbital:
        if order not in {"legacy_ijab", "spin_ijab", "ijab"}:
            raise ValueError("spin_orbital=True expects integral_order='spin_ijab' or 'legacy_ijab'")
        convention = "spin_orbital_antisymmetrized_from_ijab"
        for i in occ:
            for j in occ:
                for a_i, a in enumerate(vir):
                    for b_i, b in enumerate(vir):
                        denom = eps_arr[i] + eps_arr[j] - eps_arr[a] - eps_arr[b]
                        if abs(denom) < denominator_tol:
                            skipped += 1
                            continue
                        gijab = eri_arr[i, j, a, b] - eri_arr[i, j, b, a]
                        amplitudes[i, j, a_i, b_i] = gijab / denom
                        emp2 += 0.25 * np.conj(gijab) * gijab / denom
    else:
        if order not in {"chemist", "legacy_ijab", "ijab"}:
            raise ValueError("integral_order must be 'chemist' or 'legacy_ijab'")
        convention = "closed_shell_chemist" if order == "chemist" else "closed_shell_legacy_ijab"
        for i in occ:
            for j in occ:
                for a_i, a in enumerate(vir):
                    for b_i, b in enumerate(vir):
                        denom = eps_arr[i] + eps_arr[j] - eps_arr[a] - eps_arr[b]
                        if abs(denom) < denominator_tol:
                            skipped += 1
                            continue
                        if order == "chemist":
                            direct = eri_arr[i, a, j, b]  # (ia|jb)
                            exchange = eri_arr[i, b, j, a]  # (ib|ja)
                        else:
                            direct = eri_arr[i, j, a, b]
                            exchange = eri_arr[i, j, b, a]
                        amplitudes[i, j, a_i, b_i] = direct / denom
                        emp2 += direct * (2.0 * direct - exchange) / denom

    emp2_real = float(np.real_if_close(emp2).real)
    return MP2Result(
        correlation_energy=emp2_real,
        total_energy=float(hf_energy + emp2_real),
        amplitudes_shape=amplitudes.shape,
        skipped_denominators=skipped,
        convention=convention,
    )


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
        "absolute_kcal_per_mol_error": abs(kcal_per_mol),
        "within_chemical_accuracy": bool(abs(kcal_per_mol) <= 1.0),
    }


def state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> FidelityResult:
    a, _ = _validate_n_qubits_state(state_a)
    b, _ = _validate_n_qubits_state(state_b)
    if a.shape != b.shape:
        raise ValueError(f"state shapes differ: {a.shape} vs {b.shape}")
    amp = np.vdot(a, b)
    fid = float(np.clip(abs(amp) ** 2, 0.0, 1.0))
    return FidelityResult(fid, float(np.sqrt(max(0.0, 1.0 - fid))), amp)


def natural_orbital_occupations(one_rdm: np.ndarray, *, check_hermitian: bool = True, atol: float = 1e-10) -> np.ndarray:
    mat = np.asarray(one_rdm, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"one_rdm must be a square matrix; got shape {mat.shape}")
    if check_hermitian and not np.allclose(mat, mat.conj().T, atol=atol):
        raise ValueError("one_rdm must be Hermitian")
    vals = np.linalg.eigvalsh((mat + mat.conj().T) / 2.0)
    return np.sort(vals.real)[::-1]


def particle_number_expectation(state: np.ndarray, n_qubits: Optional[int] = None) -> float:
    vec, n_qubits = _validate_n_qubits_state(state, n_qubits)
    probs = np.abs(vec) ** 2
    exp = 0.0
    for idx, p in enumerate(probs):
        exp += float(p) * int(idx).bit_count()
    return float(exp)


def spin_z_expectation(state: np.ndarray, n_qubits: Optional[int] = None) -> float:
    """Return <S_z> assuming interleaved spin-orbital order alpha,beta,alpha,beta,..."""

    vec, n_qubits = _validate_n_qubits_state(state, n_qubits)
    probs = np.abs(vec) ** 2
    exp = 0.0
    for idx, p in enumerate(probs):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        alpha = sum(bits[0::2])
        beta = sum(bits[1::2])
        exp += float(p) * 0.5 * (alpha - beta)
    return float(exp)


def active_space_indices(
    n_orbitals: int,
    n_core: int = 0,
    n_active: Optional[int] = None,
    n_virtual: int = 0,
) -> List[int]:
    n_orb = int(n_orbitals)
    n_core = int(n_core)
    n_virtual = int(n_virtual)
    if n_orb < 0 or n_core < 0 or n_virtual < 0:
        raise ValueError("n_orbitals, n_core, and n_virtual must be non-negative")
    if n_core + n_virtual > n_orb:
        raise ValueError("n_core + n_virtual cannot exceed n_orbitals")
    if n_active is None:
        stop = n_orb - n_virtual
    else:
        n_active = int(n_active)
        if n_active < 0:
            raise ValueError("n_active must be non-negative")
        stop = n_core + n_active
        if stop > n_orb - n_virtual:
            raise ValueError("requested active space overlaps frozen virtual orbitals")
    return list(range(n_core, stop))


def freeze_core_energy_shift(
    core_orbital_energies: Sequence[float],
    *,
    nuclear_repulsion: float = 0.0,
    core_coulomb_exchange_shift: float = 0.0,
) -> float:
    """Return a lightweight frozen-core scalar energy shift.

    With only orbital energies available, the old behaviour ``2 * sum(eps_core)``
    is preserved.  Real molecular Hamiltonian generation should prefer the
    chemistry driver / integral transformation that can include the one- and
    two-electron frozen-core contributions explicitly.
    """

    eps_core = np.asarray(core_orbital_energies, dtype=float)
    return float(2.0 * np.sum(eps_core) + nuclear_repulsion + core_coulomb_exchange_shift)


def symmetry_project_state(
    state: np.ndarray,
    n_qubits: Optional[int] = None,
    n_particles: Optional[int] = None,
    sz: Optional[float] = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    vec, n_qubits = _validate_n_qubits_state(state, n_qubits)
    out = np.zeros_like(vec, dtype=np.complex128)
    for idx, amp in enumerate(vec):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if n_particles is not None and sum(bits) != int(n_particles):
            continue
        if sz is not None:
            alpha = sum(bits[0::2])
            beta = sum(bits[1::2])
            if abs(0.5 * (alpha - beta) - float(sz)) > 1e-12:
                continue
        out[idx] = amp
    norm = np.linalg.norm(out)
    if normalize and norm > 0:
        return out / norm
    return out


def commutator_norm(op_a: np.ndarray, op_b: np.ndarray) -> float:
    a = np.asarray(op_a, dtype=np.complex128)
    b = np.asarray(op_b, dtype=np.complex128)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1] or a.shape != b.shape:
        raise ValueError(f"op_a and op_b must be square matrices with the same shape; got {a.shape} and {b.shape}")
    c = a @ b - b @ a
    return float(np.linalg.norm(c))
