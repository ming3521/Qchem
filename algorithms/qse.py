"""Quantum Subspace Expansion (QSE).

This module provides a small classical reference implementation of QSE.  Given a
reference state |psi> and a list of excitation operators O_mu, it builds the
subspace spanned by |psi> and O_mu|psi>, computes

    H_mu,nu = <phi_mu|H|phi_nu>,    S_mu,nu = <phi_mu|phi_nu>,

and solves the generalized eigenvalue problem H c = E S c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .classical_eigensolvers import hamiltonian_matrix, normalize_pauli_terms


@dataclass
class QSEResult:
    energies: np.ndarray
    coefficients: np.ndarray
    overlap_matrix: np.ndarray
    hamiltonian_matrix: np.ndarray
    regularization: float
    overlap_eigenvalues: Optional[np.ndarray] = None
    kept_subspace_dimension: Optional[int] = None
    basis_vectors: Optional[np.ndarray] = None


def _as_state_vector(state: np.ndarray, n_qubits: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Validate and normalize a state vector."""

    vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vec.size == 0:
        raise ValueError("reference_state must be a non-empty state vector")

    inferred = int(round(np.log2(vec.size)))
    if 2**inferred != vec.size:
        raise ValueError(
            f"reference_state length must be a power of 2, got length {vec.size}"
        )

    if n_qubits is None:
        n_qubits = inferred
    else:
        n_qubits = int(n_qubits)
        if n_qubits < 0:
            raise ValueError("n_qubits must be non-negative")
        if 2**n_qubits != vec.size:
            raise ValueError(
                f"reference_state length {vec.size} is incompatible with n_qubits={n_qubits}"
            )

    norm = np.linalg.norm(vec)
    if norm <= 0:
        raise ValueError("reference_state must have non-zero norm")
    return vec / norm, n_qubits


def _operator_to_hamiltonian_like(operator):
    """Accept a few convenient single-operator forms.

    The library standard Hamiltonian format is ``[(coeff, 'PauliString')]``.
    For QSE excitation pools it is also convenient to accept ``'XI'`` or
    ``(coeff, 'XI')`` as a shorthand for a one-term operator.
    """

    if isinstance(operator, str):
        return [(1.0, operator)]

    if isinstance(operator, tuple) and len(operator) == 2:
        coeff, pauli = operator
        if isinstance(pauli, str) and np.isscalar(coeff):
            return [(coeff, pauli)]

    return operator


def apply_operator_matrix(operator: np.ndarray, state: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Apply a dense operator to a state vector.

    Args:
        operator: Dense matrix with shape ``(dim, dim)``.
        state: State vector with shape ``(dim,)``.
        normalize: Whether to normalize the output when it is non-zero.
    """

    op = np.asarray(operator, dtype=np.complex128)
    vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError(f"operator must be a square matrix, got shape {op.shape}")
    if op.shape[1] != vec.size:
        raise ValueError(
            f"operator shape {op.shape} is incompatible with state length {vec.size}"
        )

    out = op @ vec
    if not normalize:
        return out
    norm = np.linalg.norm(out)
    if norm == 0:
        return out
    return out / norm


def build_excitation_matrices(excitation_terms, n_qubits: int) -> List[np.ndarray]:
    """Convert QSE excitation operators to dense matrices.

    Each entry can be any format accepted by ``hamiltonian_matrix``.  In addition,
    a bare Pauli string such as ``'XI'`` or a single term ``(1.0, 'XI')`` is
    accepted as shorthand.
    """

    if excitation_terms is None:
        return []

    mats: List[np.ndarray] = []
    for op in excitation_terms:
        mat = hamiltonian_matrix(
            _operator_to_hamiltonian_like(op), n_qubits=n_qubits, sparse=False
        )
        mats.append(np.asarray(mat, dtype=np.complex128))
    return mats


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _solve_qse_generalized_eigenproblem(
    heff: np.ndarray, overlap: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve H c = E S c using canonical orthogonalization.

    This is more stable than solving against ``S + eps I`` because it removes
    directions whose overlap eigenvalues are numerically zero instead of turning
    them into artificial physical states.
    """

    heff = _symmetrize(np.asarray(heff, dtype=np.complex128))
    overlap = _symmetrize(np.asarray(overlap, dtype=np.complex128))

    s_vals, s_vecs = np.linalg.eigh(overlap)
    max_s = float(np.max(np.abs(s_vals))) if s_vals.size else 0.0
    threshold = max(float(cutoff), float(cutoff) * max(1.0, max_s))
    keep = s_vals > threshold

    if not np.any(keep):
        raise ValueError(
            "All QSE basis directions were removed by the overlap cutoff; "
            "try lowering regularization/overlap_cutoff or changing the excitation pool."
        )

    # Columns of transform map reduced orthonormal coordinates back to the
    # original non-orthogonal normalized basis.
    transform = s_vecs[:, keep] / np.sqrt(s_vals[keep])[None, :]
    h_orth = _symmetrize(transform.conj().T @ heff @ transform)
    energies, reduced_coeffs = np.linalg.eigh(h_orth)
    coeffs = transform @ reduced_coeffs
    return energies.real, coeffs, s_vals.real, int(np.count_nonzero(keep))


def quantum_subspace_expansion(
    reference_state: np.ndarray,
    hamiltonian,
    excitation_terms,
    n_qubits: Optional[int] = None,
    regularization: float = 1e-8,
    overlap_cutoff: Optional[float] = None,
    hamiltonian_hermitian_tol: float = 1e-9,
) -> QSEResult:
    """Solve the Quantum Subspace Expansion generalized eigenvalue problem.

    ``excitation_terms`` is a list of Pauli-term Hamiltonian-like operators.  The
    normalized reference state itself is included automatically as the first
    subspace vector.

    Args:
        reference_state: State vector ``|psi>``.
        hamiltonian: Hamiltonian in the same Pauli format accepted by
            ``classical_eigensolvers.hamiltonian_matrix``.
        excitation_terms: Iterable of excitation operators.  Each operator may be
            ``[(coeff, 'PauliString')]``, a bare ``'PauliString'``, or another
            format accepted by ``hamiltonian_matrix``.
        n_qubits: Optional number of qubits.  Inferred from ``reference_state``
            when omitted.
        regularization: Backward-compatible cutoff scale.  It is used to skip
            zero vectors and, when ``overlap_cutoff`` is omitted, to remove
            linearly dependent directions in the overlap matrix.
        overlap_cutoff: Optional explicit cutoff for overlap eigenvalues.
        hamiltonian_hermitian_tol: Tolerance used to validate Hermiticity of H.
    """

    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    cutoff = regularization if overlap_cutoff is None else float(overlap_cutoff)
    if cutoff < 0:
        raise ValueError("overlap_cutoff must be non-negative")

    ref, n_qubits = _as_state_vector(reference_state, n_qubits)

    h_terms = normalize_pauli_terms(hamiltonian)
    H = np.asarray(hamiltonian_matrix(h_terms, n_qubits=n_qubits, sparse=False), dtype=np.complex128)
    if not np.allclose(H, H.conj().T, atol=hamiltonian_hermitian_tol):
        raise ValueError("QSE requires a Hermitian Hamiltonian matrix")
    H = _symmetrize(H)

    basis = [ref]
    for mat in build_excitation_matrices(excitation_terms, n_qubits):
        if mat.shape != H.shape:
            raise ValueError(
                f"excitation operator shape {mat.shape} does not match Hamiltonian shape {H.shape}"
            )
        vec = mat @ ref
        norm = np.linalg.norm(vec)
        if norm > regularization:
            basis.append(vec / norm)

    basis_matrix = np.column_stack(basis)
    overlap = basis_matrix.conj().T @ basis_matrix
    heff = basis_matrix.conj().T @ H @ basis_matrix
    overlap = _symmetrize(overlap)
    heff = _symmetrize(heff)

    energies, coeffs, s_vals, kept_dim = _solve_qse_generalized_eigenproblem(
        heff, overlap, cutoff
    )

    return QSEResult(
        energies=energies,
        coefficients=coeffs,
        overlap_matrix=overlap,
        hamiltonian_matrix=heff,
        regularization=regularization,
        overlap_eigenvalues=s_vals,
        kept_subspace_dimension=kept_dim,
        basis_vectors=basis_matrix,
    )


def _pauli_word(n_qubits: int, ops: dict[int, str]) -> str:
    chars = ["I"] * n_qubits
    for q, p in ops.items():
        q = int(q)
        if q < 0 or q >= n_qubits:
            raise IndexError(f"Qubit index {q} out of range [0, {n_qubits - 1}]")
        p = str(p).upper()
        if p not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"Unsupported Pauli '{p}'")
        chars[q] = p
    return "".join(chars)


def singles_doubles_qse_pool(n_qubits: int, n_electrons: int):
    """Generate a simple singles/doubles Pauli pool for QSE diagnostics.

    The returned operators use the library's standard Hamiltonian format, for
    example ``[(0.5, 'XIXI'), (0.5, 'YIYI')]``.
    """

    n_qubits = int(n_qubits)
    n_electrons = int(n_electrons)
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_electrons < 0 or n_electrons > n_qubits:
        raise ValueError("n_electrons must satisfy 0 <= n_electrons <= n_qubits")

    pool = []
    occ = range(n_electrons)
    vir = range(n_electrons, n_qubits)

    for i in occ:
        for a in vir:
            pool.append(
                [
                    (0.5, _pauli_word(n_qubits, {i: "X", a: "X"})),
                    (0.5, _pauli_word(n_qubits, {i: "Y", a: "Y"})),
                ]
            )

    for i in occ:
        for j in occ:
            if j <= i:
                continue
            for a in vir:
                for b in vir:
                    if b <= a:
                        continue
                    pool.append(
                        [
                            (
                                0.25,
                                _pauli_word(
                                    n_qubits,
                                    {i: "X", j: "X", a: "X", b: "X"},
                                ),
                            )
                        ]
                    )

    return pool
