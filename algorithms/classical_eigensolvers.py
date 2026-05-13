"""Classical eigensolvers used as references for quantum chemistry algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from backends._matrix_simulator import apply_operator, gate_matrix, zero_state

PauliString = Tuple[Tuple[int, str], ...]


@dataclass
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray]
    method: str
    converged: bool = True
    n_iterations: Optional[int] = None


_VALID_PAULIS = {"I", "X", "Y", "Z"}


def _load_scipy_sparse():
    """Import scipy sparse modules only when the sparse path is requested.

    Keeping SciPy out of module import makes this file safer in notebook
    environments where a broken SciPy/OpenBLAS/ARPACK build can crash the kernel.
    """

    try:
        import scipy.sparse as scipy_sparse
        import scipy.sparse.linalg as scipy_sparse_linalg
    except Exception:  # pragma: no cover - optional dependency guard
        return None, None
    return scipy_sparse, scipy_sparse_linalg


def _is_scalar_like(value) -> bool:
    """Return True when *value* can be used as a numeric coefficient."""

    if isinstance(value, str):
        try:
            complex(value)
            return True
        except ValueError:
            return False
    try:
        complex(value)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_pauli_label(label: str) -> str:
    pauli = str(label).upper()
    if pauli not in _VALID_PAULIS:
        raise ValueError(f"Unknown Pauli '{label}'")
    return pauli


def _pauli_word_to_term(pauli_word: str) -> PauliString:
    """Convert a compact Pauli word, e.g. ``"ZI"``, to a PauliString.

    The library's solver examples use Hamiltonians of the form
    ``[(coefficient, pauli_word), ...]``.  Qubit 0 is the left-most character,
    matching the state-vector convention used by ``backends._matrix_simulator``.

    For convenience this also accepts OpenFermion-style strings such as
    ``"Z0 X1"``.
    """

    text = pauli_word.strip().upper()
    if not text:
        return tuple()

    tokens = text.replace("*", " ").split()
    # Indexed forms: "Z0 X1" or a single token like "Z0".
    if len(tokens) > 1 or any(ch.isdigit() for ch in text):
        parsed = []
        for token in tokens:
            pauli = _normalize_pauli_label(token[0])
            if len(token) == 1:
                raise ValueError(
                    f"Ambiguous Pauli token '{token}'. Use compact form like 'ZI' "
                    "or indexed form like 'Z0 X1'."
                )
            parsed.append((int(token[1:]), pauli))
        return tuple(parsed)

    if any(ch not in _VALID_PAULIS for ch in text):
        raise ValueError(
            f"Invalid compact Pauli word '{pauli_word}'. Allowed characters are I, X, Y, Z."
        )
    # Keep explicit I terms so infer_n_qubits can recover the intended length,
    # e.g. [(1.0, "II")] should be a 2-qubit identity when n_qubits is omitted.
    return tuple((q, ch) for q, ch in enumerate(text))


def _term_like_to_pauli_string(term) -> PauliString:
    """Normalize a term spec into ``((qubit, pauli), ...)`` form."""

    if term is None:
        return tuple()
    if isinstance(term, str):
        return _pauli_word_to_term(term)

    term_tuple = tuple(term)
    if not term_tuple:
        return tuple()

    parsed = []
    for item in term_tuple:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "Pauli terms must be compact strings like 'ZI' or iterables of "
                "(qubit, pauli) pairs."
            )
        qubit, pauli = item
        parsed.append((int(qubit), _normalize_pauli_label(pauli)))
    return tuple(parsed)


def _normalize_mapping_items(items):
    return [(_term_like_to_pauli_string(k), complex(v)) for k, v in items]


def normalize_pauli_terms(hamiltonian):
    """Normalize supported Hamiltonian formats to ``[(PauliString, coeff), ...]``.

    Supported inputs include:

    - Library-standard format: ``[(coeff, "ZI"), (coeff, "XX")]``.
    - Mapping standard format: ``{"ZI": coeff, "XX": coeff}``.
    - OpenFermion-like objects with ``.terms`` mapping
      ``{((0, "Z"),): coeff}``.
    - Backward-compatible format: ``[((0, "Z"),), coeff]`` per term, i.e.
      ``[(PauliString, coeff), ...]``.
    """

    if hasattr(hamiltonian, "terms"):
        return _normalize_mapping_items(hamiltonian.terms.items())

    if isinstance(hamiltonian, Mapping):
        return _normalize_mapping_items(hamiltonian.items())

    normalized = []
    for item in hamiltonian:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "Hamiltonian must contain 2-tuples in either (coeff, pauli_word) "
                "or (pauli_term, coeff) form."
            )
        first, second = item

        if _is_scalar_like(first) and not _is_scalar_like(second):
            coeff, term = first, second
        elif not _is_scalar_like(first) and _is_scalar_like(second):
            term, coeff = first, second
        elif _is_scalar_like(first) and isinstance(second, str):
            coeff, term = first, second
        else:
            raise ValueError(
                "Could not infer Hamiltonian term order. Use (coeff, 'ZI') for "
                "the library-standard format or (((0, 'Z'),), coeff) for "
                "OpenFermion-like terms."
            )

        normalized.append((_term_like_to_pauli_string(term), complex(coeff)))
    return normalized


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
        sp, _ = _load_scipy_sparse()
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
        sp, _ = _load_scipy_sparse()
        if sp is None:
            raise ImportError("scipy is required for sparse matrices")
        mat = sp.csr_matrix((dim, dim), dtype=np.complex128)
    else:
        sp = None
        mat = np.zeros((dim, dim), dtype=np.complex128)
    for term, coeff in terms:
        if not term:
            ident = sp.eye(dim, format="csr") if sparse else np.eye(dim, dtype=np.complex128)
            mat = mat + coeff * ident
        else:
            mat = mat + coeff * pauli_string_matrix(term, n_qubits, sparse=sparse)
    return mat


def exact_diagonalization(
    hamiltonian,
    n_qubits: Optional[int] = None,
    k: Optional[int] = None,
    return_vectors: bool = True,
    *,
    sparse_threshold: int = 128,
    use_sparse: bool = True,
) -> EigenResult:
    """Diagonalize a Pauli Hamiltonian.

    Dense ``numpy.linalg.eigh`` is used by default for small Hilbert spaces.
    Sparse ``scipy.sparse.linalg.eigsh`` is only attempted when ``dim`` is at
    least ``sparse_threshold``.  This avoids ARPACK overhead and avoids kernel
    crashes seen in some notebook environments for tiny matrices.
    """

    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = infer_n_qubits(terms)
    dim = 2**n_qubits

    can_use_sparse = use_sparse and k is not None and dim >= sparse_threshold and k < dim - 1
    if can_use_sparse:
        sp, spla = _load_scipy_sparse()
        if sp is not None and spla is not None:
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


def lanczos_lowest_eigenvalue(
    hamiltonian,
    n_qubits: Optional[int] = None,
    maxiter: int = 80,
    tol: float = 1e-10,
    seed: Optional[int] = None,
    *,
    sparse_threshold: int = 128,
    use_scipy: bool = True,
) -> EigenResult:
    """Return the lowest eigenvalue using a safe dense fallback for small systems."""

    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = infer_n_qubits(terms)
    dim = 2**n_qubits

    if dim < sparse_threshold or not use_scipy:
        mat = hamiltonian_matrix(terms, n_qubits, sparse=False)
        vals, vecs = np.linalg.eigh(mat)
        return EigenResult(vals[:1], vecs[:, :1], method="dense-eigh-lowest", n_iterations=None)

    sp, spla = _load_scipy_sparse()
    if sp is not None and spla is not None:
        mat = hamiltonian_matrix(terms, n_qubits, sparse=True)
        vals, vecs = spla.eigsh(mat, k=1, which="SA", maxiter=maxiter, tol=tol)
        idx = np.argsort(vals)
        return EigenResult(vals[idx], vecs[:, idx], method="lanczos-scipy", n_iterations=maxiter)

    rng = np.random.default_rng(seed)
    q = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    q = q / np.linalg.norm(q)
    alpha = []
    beta = []
    q_prev = np.zeros_like(q)
    b = 0.0
    dense = hamiltonian_matrix(terms, n_qubits, sparse=False)
    for _ in range(maxiter):
        z = dense @ q
        a = np.vdot(q, z).real
        z = z - a * q - b * q_prev
        b = np.linalg.norm(z)
        alpha.append(a)
        if b < tol:
            break
        beta.append(b)
        q_prev = q
        q = z / b
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    vals, _ = np.linalg.eigh(T)
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
