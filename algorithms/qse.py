"""Quantum Subspace Expansion (QSE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .classical_eigensolvers import hamiltonian_matrix, normalize_pauli_terms


@dataclass
class QSEResult:
    energies: np.ndarray
    coefficients: np.ndarray
    overlap_matrix: np.ndarray
    hamiltonian_matrix: np.ndarray
    regularization: float


def apply_operator_matrix(operator: np.ndarray, state: np.ndarray) -> np.ndarray:
    out = operator @ state
    norm = np.linalg.norm(out)
    if norm == 0:
        return out
    return out / norm


def build_excitation_matrices(excitation_terms, n_qubits: int) -> List[np.ndarray]:
    mats = []
    for op in excitation_terms:
        mat = hamiltonian_matrix(op, n_qubits=n_qubits, sparse=False)
        mats.append(mat)
    return mats


def quantum_subspace_expansion(reference_state: np.ndarray, hamiltonian, excitation_terms, n_qubits: Optional[int] = None, regularization: float = 1e-8) -> QSEResult:
    """Solve the QSE generalized eigenvalue problem.

    ``excitation_terms`` is a list of Pauli-term Hamiltonian-like operators.  The
    reference state itself is included automatically as the first basis vector.
    """

    h_terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = int(np.log2(reference_state.size))
    H = hamiltonian_matrix(h_terms, n_qubits=n_qubits, sparse=False)
    basis = [reference_state / np.linalg.norm(reference_state)]
    for mat in build_excitation_matrices(excitation_terms, n_qubits):
        vec = mat @ reference_state
        if np.linalg.norm(vec) > regularization:
            basis.append(vec / np.linalg.norm(vec))
    m = len(basis)
    S = np.zeros((m, m), dtype=np.complex128)
    Heff = np.zeros((m, m), dtype=np.complex128)
    for i, vi in enumerate(basis):
        for j, vj in enumerate(basis):
            S[i, j] = np.vdot(vi, vj)
            Heff[i, j] = np.vdot(vi, H @ vj)
    S_reg = S + regularization * np.eye(m)
    try:
        import scipy.linalg

        vals, coeffs = scipy.linalg.eigh(Heff, S_reg)
    except Exception:
        vals, coeffs = np.linalg.eig(np.linalg.solve(S_reg, Heff))
        idx = np.argsort(vals.real)
        vals = vals[idx]
        coeffs = coeffs[:, idx]
    return QSEResult(vals.real, coeffs, S, Heff, regularization)


def singles_doubles_qse_pool(n_qubits: int, n_electrons: int):
    """Generate a simple Pauli pool suitable for QSE diagnostics."""

    pool = []
    occ = range(n_electrons)
    vir = range(n_electrons, n_qubits)
    for i in occ:
        for a in vir:
            pool.append([(((i, "X"), (a, "X")), 0.5), (((i, "Y"), (a, "Y")), 0.5)])
    for i in occ:
        for j in occ:
            if j <= i:
                continue
            for a in vir:
                for b in vir:
                    if b <= a:
                        continue
                    pool.append([(((i, "X"), (j, "X"), (a, "X"), (b, "X")), 0.25)])
    return pool
