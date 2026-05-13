"""Classical eigensolvers used as references for quantum chemistry algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.linalg
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover
    scipy = None  # type: ignore
    sp = None  # type: ignore
    spla = None  # type: ignore

from backends._matrix_simulator import apply_operator, gate_matrix, zero_state

PauliString = Tuple[Tuple[int, str], ...]


@dataclass
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray]
    method: str
    converged: bool = True
    n_iterations: Optional[int] = None


def normalize_pauli_terms(hamiltonian):
    if hasattr(hamiltonian, "terms"):
        return [(tuple(k), complex(v)) for k, v in hamiltonian.terms.items()]
    if isinstance(hamiltonian, Mapping):
        return [(tuple(k), complex(v)) for k, v in hamiltonian.items()]
    return [(tuple(k), complex(v)) for k, v in hamiltonian]


def infer_n_qubits(terms) -> int:
    n = 0
    for term, _ in terms:
        for q, _ in term:
            n = max(n, int(q) + 1)
    return n


def pauli_matrix(pauli: str) -> np.ndarray:
    p = str(pauli).upper()
    if p == "I":
        return np.eye(2, dtype=np.complex128)
    mat = gate_matrix(p, [])
    if mat is None:
        raise ValueError(f"Unknown Pauli '{pauli}'")
    return mat


def pauli_string_matrix(term: PauliString, n_qubits: int, sparse: bool = False):
    mats = []
    term_map = {int(q): str(p).upper() for q, p in term}
    for q in range(n_qubits):
        mats.append(pauli_matrix(term_map.get(q, "I")))
    if sparse:
        if sp is None:
            raise ImportError("scipy is required for sparse matrices")
        result = sp.csr_matrix([[1.0 + 0.0j]])
        for mat in mats:
            result = sp.kron(result, sp.csr_matrix(mat), format="csr")
        return result
    result = mats[0]
    for mat in mats[1:]:
        result = np.kron(result, mat)
    return result


def hamiltonian_matrix(hamiltonian, n_qubits: Optional[int] = None, sparse: bool = False):
    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = infer_n_qubits(terms)
    dim = 2**n_qubits
    if sparse:
        if sp is None:
            raise ImportError("scipy is required for sparse matrices")
        mat = sp.csr_matrix((dim, dim), dtype=np.complex128)
    else:
        mat = np.zeros((dim, dim), dtype=np.complex128)
    for term, coeff in terms:
        if not term:
            mat = mat + coeff * (sp.eye(dim, format="csr") if sparse else np.eye(dim, dtype=np.complex128))
        else:
            mat = mat + coeff * pauli_string_matrix(term, n_qubits, sparse=sparse)
    return mat


def exact_diagonalization(hamiltonian, n_qubits: Optional[int] = None, k: Optional[int] = None, return_vectors: bool = True) -> EigenResult:
    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = infer_n_qubits(terms)
    dim = 2**n_qubits
    if k is not None and k < dim - 1 and sp is not None:
        mat = hamiltonian_matrix(terms, n_qubits, sparse=True)
        vals, vecs = spla.eigsh(mat, k=k, which="SA")
        idx = np.argsort(vals)
        return EigenResult(vals[idx], vecs[:, idx] if return_vectors else None, method="sparse-eigsh")
    mat = hamiltonian_matrix(terms, n_qubits, sparse=False)
    vals, vecs = np.linalg.eigh(mat)
    if k is not None:
        vals = vals[:k]
        vecs = vecs[:, :k]
    return EigenResult(vals, vecs if return_vectors else None, method="dense-eigh")


def lanczos_lowest_eigenvalue(hamiltonian, n_qubits: Optional[int] = None, maxiter: int = 80, tol: float = 1e-10, seed: Optional[int] = None) -> EigenResult:
    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = infer_n_qubits(terms)
    mat = hamiltonian_matrix(terms, n_qubits, sparse=True if sp is not None else False)
    if spla is not None:
        vals, vecs = spla.eigsh(mat, k=1, which="SA", maxiter=maxiter, tol=tol)
        return EigenResult(vals, vecs, method="lanczos-scipy", n_iterations=maxiter)
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    q = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    q = q / np.linalg.norm(q)
    Q = []
    alpha = []
    beta = []
    q_prev = np.zeros_like(q)
    b = 0.0
    dense = np.asarray(mat)
    for it in range(maxiter):
        z = dense @ q
        a = np.vdot(q, z).real
        z = z - a * q - b * q_prev
        b = np.linalg.norm(z)
        Q.append(q)
        alpha.append(a)
        if b < tol:
            break
        beta.append(b)
        q_prev = q
        q = z / b
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    vals, vecs = np.linalg.eigh(T)
    return EigenResult(vals[:1], None, method="lanczos-numpy", converged=b < tol, n_iterations=len(alpha))


def rayleigh_quotient(matrix: np.ndarray, vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.complex128)
    return float(np.vdot(vector, matrix @ vector).real / np.vdot(vector, vector).real)


def reduced_density_matrix(state: np.ndarray, keep: Sequence[int], n_qubits: int) -> np.ndarray:
    keep = list(keep)
    trace = [q for q in range(n_qubits) if q not in keep]
    tensor = state.reshape([2] * n_qubits)
    perm = keep + trace
    tensor = np.transpose(tensor, perm)
    dim_keep = 2 ** len(keep)
    dim_trace = 2 ** len(trace)
    psi = tensor.reshape(dim_keep, dim_trace)
    return psi @ psi.conj().T


def hartree_fock_bitstring(n_qubits: int, n_electrons: int) -> str:
    return "".join("1" if i < n_electrons else "0" for i in range(n_qubits))


def hartree_fock_state(n_qubits: int, n_electrons: int) -> np.ndarray:
    index = 0
    for q in range(n_qubits):
        index = (index << 1) | (1 if q < n_electrons else 0)
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[index] = 1.0
    return state
