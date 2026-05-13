# Qchem Extension Complete Code

This file concatenates all generated extension files. Copy files into the corresponding folders of the Qchem repository.

## README_EXTENSION.md

```
# Qchem extension code for software-copyright code expansion

This directory contains drop-in Python files for the public `ming3521/Qchem`
repository.  The extension follows the existing folder layout:

- `backends/`: adds `CirqBackend`, `QulacsBackend`, `QutipBackend` and shared
  dense simulation utilities.
- `ansatz/`: adds `RyRzAnsatz`, compact pair-excitation / k-UpCCG ansatz, and an
  ADAPT-VQE style adaptive ansatz toolkit.
- `algorithms/`: adds Trotter simulation, phase estimation, classical exact and
  Lanczos eigensolvers, QSE, VQD, MP2/FCI/fidelity/symmetry utilities.

Suggested copy command from the repository root:

```bash
cp -r /path/to/qchem_extension/backends/*.py backends/
cp -r /path/to/qchem_extension/ansatz/*.py ansatz/
cp -r /path/to/qchem_extension/algorithms/*.py algorithms/
```

Optional dependencies:

```bash
pip install cirq qulacs qutip qutip-qip scipy
```

The code is designed so optional backend packages are imported only when those
backends are actually instantiated.
```

## algorithms/__init__.py

```python
try:
    from .shadow import *  # noqa: F401,F403
except Exception:
    pass

from .trotter import append_pauli_evolution, first_order_trotter_circuit, second_order_trotter_circuit, trotter_info
from .classical_eigensolvers import exact_diagonalization, lanczos_lowest_eigenvalue, hamiltonian_matrix, hartree_fock_state
from .phase_estimation import build_qpe_circuit, phases_from_counts, exact_phase_estimation_from_unitary
from .qse import quantum_subspace_expansion, singles_doubles_qse_pool
from .vqd import vqd_objective, deflated_matrix, gram_schmidt_states
from .quantum_chemistry import (
    mp2_energy_from_integrals,
    fci_reference_energy,
    chemical_accuracy_error,
    state_fidelity,
    particle_number_expectation,
    spin_z_expectation,
    symmetry_project_state,
)

__all__ = [
    "append_pauli_evolution",
    "first_order_trotter_circuit",
    "second_order_trotter_circuit",
    "trotter_info",
    "exact_diagonalization",
    "lanczos_lowest_eigenvalue",
    "hamiltonian_matrix",
    "hartree_fock_state",
    "build_qpe_circuit",
    "phases_from_counts",
    "exact_phase_estimation_from_unitary",
    "quantum_subspace_expansion",
    "singles_doubles_qse_pool",
    "vqd_objective",
    "deflated_matrix",
    "gram_schmidt_states",
    "mp2_energy_from_integrals",
    "fci_reference_energy",
    "chemical_accuracy_error",
    "state_fidelity",
    "particle_number_expectation",
    "spin_z_expectation",
    "symmetry_project_state",
]
```

## algorithms/classical_eigensolvers.py

```python
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
```

## algorithms/phase_estimation.py

```python
"""Quantum phase estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from backends.core import QuantumCircuit


@dataclass
class PhaseEstimationResult:
    phases: np.ndarray
    probabilities: np.ndarray
    energies: Optional[np.ndarray] = None
    bitstrings: Optional[List[str]] = None


def inverse_qft(qc: QuantumCircuit, qubits: Sequence[int], do_swaps: bool = True):
    qubits = list(qubits)
    n = len(qubits)
    if do_swaps:
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
    for j in range(n):
        q = qubits[j]
        for m in range(j):
            angle = -np.pi / (2 ** (j - m))
            qc.cp(qubits[m], q, angle)
        qc.h(q)


def qft(qc: QuantumCircuit, qubits: Sequence[int], do_swaps: bool = True):
    qubits = list(qubits)
    n = len(qubits)
    for j in reversed(range(n)):
        q = qubits[j]
        qc.h(q)
        for m in reversed(range(j)):
            angle = np.pi / (2 ** (j - m))
            qc.cp(qubits[m], q, angle)
    if do_swaps:
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])


def build_qpe_circuit(unitary_circuit: QuantumCircuit, n_ancilla: int, target_qubits: Optional[Sequence[int]] = None) -> QuantumCircuit:
    """Build standard QPE for a unitary represented as a HYQ sub-circuit.

    The current ``QuantumCircuit`` supports circuit extension but not arbitrary
    controlled sub-circuit synthesis in every backend.  This builder therefore
    appends controlled copies of each instruction in the supplied unitary circuit.
    """

    if target_qubits is None:
        target_qubits = list(range(n_ancilla, n_ancilla + unitary_circuit.n_qubits))
    target_qubits = list(target_qubits)
    qc = QuantumCircuit(n_ancilla + unitary_circuit.n_qubits, name="QPE")
    anc = list(range(n_ancilla))
    qc.h(anc)
    for k, a in enumerate(anc):
        repeats = 2**k
        for _ in range(repeats):
            for inst in unitary_circuit.instructions:
                copied = qc.append(inst.name, [target_qubits[q] for q in inst.qubits], params=list(inst.params))
                copied.control(a)
    inverse_qft(qc, anc)
    return qc


def phases_from_counts(counts, n_ancilla: int) -> PhaseEstimationResult:
    total = sum(counts.values())
    bitstrings = sorted(counts, key=counts.get, reverse=True)
    probs = np.array([counts[b] / total for b in bitstrings], dtype=float)
    phases = np.array([int(b, 2) / (2**n_ancilla) for b in bitstrings], dtype=float)
    return PhaseEstimationResult(phases=phases, probabilities=probs, bitstrings=bitstrings)


def exact_phase_estimation_from_unitary(unitary: np.ndarray, input_state: np.ndarray, n_ancilla: int, energy_scale: Optional[float] = None) -> PhaseEstimationResult:
    """Classical reference for QPE output probabilities."""

    vals, vecs = np.linalg.eig(unitary)
    phases = (np.angle(vals) / (2 * np.pi)) % 1.0
    overlaps = np.abs(vecs.conj().T @ input_state) ** 2
    grid = np.arange(2**n_ancilla) / (2**n_ancilla)
    probs = np.zeros_like(grid, dtype=float)
    for phase, weight in zip(phases, overlaps):
        idx = int(np.round(phase * 2**n_ancilla)) % (2**n_ancilla)
        probs[idx] += weight.real
    energies = None
    if energy_scale is not None:
        energies = 2 * np.pi * grid / energy_scale
    return PhaseEstimationResult(phases=grid, probabilities=probs, energies=energies, bitstrings=[format(i, f"0{n_ancilla}b") for i in range(2**n_ancilla)])


def iterative_phase_estimation_step(expectation_cos: float, expectation_sin: float) -> float:
    """Return a phase estimate from Hadamard-test cosine/sine estimates."""

    return float(np.arctan2(expectation_sin, expectation_cos) / (2 * np.pi) % 1.0)
```

## algorithms/qse.py

```python
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
```

## algorithms/quantum_chemistry.py

```python
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
```

## algorithms/trotter.py

```python
"""Hamiltonian simulation and Trotterization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backends.core import QuantumCircuit

PauliString = Tuple[Tuple[int, str], ...]
PauliTerm = Tuple[PauliString, complex]


@dataclass
class TrotterStepInfo:
    order: int
    time: float
    n_steps: int
    n_terms: int
    gate_count_estimate: int


def normalize_pauli_terms(hamiltonian) -> List[PauliTerm]:
    """Normalize common Hamiltonian representations to ``[(term, coeff), ...]``."""

    if hasattr(hamiltonian, "terms"):
        return [(tuple(term), complex(coeff)) for term, coeff in hamiltonian.terms.items()]
    if isinstance(hamiltonian, Mapping):
        return [(tuple(term), complex(coeff)) for term, coeff in hamiltonian.items()]
    out = []
    for item in hamiltonian:
        if len(item) != 2:
            raise ValueError("Hamiltonian entries must be (pauli_string, coefficient)")
        term, coeff = item
        out.append((tuple(term), complex(coeff)))
    return out


def append_pauli_evolution(qc: QuantumCircuit, pauli_string: PauliString, angle):
    """Append exp(-i angle P) for a Pauli string P."""

    if not pauli_string:
        qc.global_phase(-angle)
        return
    qubits = [int(q) for q, _ in pauli_string]
    ops = [str(p).upper() for _, p in pauli_string]
    for q, op in zip(qubits, ops):
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.rx(q, np.pi / 2)
        elif op == "Z":
            pass
        else:
            raise ValueError(f"Unsupported Pauli operator '{op}'")
    for a, b in zip(qubits[:-1], qubits[1:]):
        qc.cx(a, b)
    qc.rz(qubits[-1], 2.0 * angle)
    for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
        qc.cx(a, b)
    for q, op in reversed(list(zip(qubits, ops))):
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.rx(q, -np.pi / 2)


def first_order_trotter_circuit(hamiltonian, time: float, n_steps: int, n_qubits: Optional[int] = None) -> QuantumCircuit:
    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = 0
        for term, _ in terms:
            for q, _p in term:
                n_qubits = max(n_qubits, int(q) + 1)
    qc = QuantumCircuit(n_qubits, name="FirstOrderTrotter")
    dt = float(time) / int(n_steps)
    for _ in range(int(n_steps)):
        for term, coeff in terms:
            if abs(coeff.imag) > 1e-10:
                raise ValueError("Trotter evolution expects real Hamiltonian coefficients")
            append_pauli_evolution(qc, term, dt * coeff.real)
    return qc


def second_order_trotter_circuit(hamiltonian, time: float, n_steps: int, n_qubits: Optional[int] = None) -> QuantumCircuit:
    terms = normalize_pauli_terms(hamiltonian)
    if n_qubits is None:
        n_qubits = max((int(q) + 1 for term, _ in terms for q, _p in term), default=0)
    qc = QuantumCircuit(n_qubits, name="SecondOrderTrotter")
    dt = float(time) / int(n_steps)
    for _ in range(int(n_steps)):
        for term, coeff in terms:
            append_pauli_evolution(qc, term, 0.5 * dt * coeff.real)
        for term, coeff in reversed(terms):
            append_pauli_evolution(qc, term, 0.5 * dt * coeff.real)
    return qc


def trotter_info(hamiltonian, time: float, n_steps: int, order: int = 1) -> TrotterStepInfo:
    terms = normalize_pauli_terms(hamiltonian)
    estimated = 0
    for term, _ in terms:
        k = len(term)
        estimated += 1 if k == 0 else 2 * max(k - 1, 0) + 1 + 2 * sum(p != "Z" for _, p in term)
    multiplier = n_steps if order == 1 else 2 * n_steps
    return TrotterStepInfo(order=order, time=float(time), n_steps=int(n_steps), n_terms=len(terms), gate_count_estimate=estimated * multiplier)
```

## algorithms/vqd.py

```python
"""Variational Quantum Deflation helpers for excited states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class VQDObjectiveBreakdown:
    energy: float
    penalties: List[float]
    total: float


def state_overlap(state_a: np.ndarray, state_b: np.ndarray) -> float:
    return float(abs(np.vdot(state_a, state_b)) ** 2)


def vqd_objective(energy: float, candidate_state: np.ndarray, previous_states: Sequence[np.ndarray], betas: Optional[Sequence[float]] = None) -> VQDObjectiveBreakdown:
    if betas is None:
        betas = [1.0] * len(previous_states)
    penalties = [float(beta) * state_overlap(candidate_state, state) for beta, state in zip(betas, previous_states)]
    total = float(energy + sum(penalties))
    return VQDObjectiveBreakdown(float(energy), penalties, total)


def gram_schmidt_states(states: Sequence[np.ndarray], atol: float = 1e-12) -> List[np.ndarray]:
    basis: List[np.ndarray] = []
    for state in states:
        v = np.asarray(state, dtype=np.complex128).copy()
        for b in basis:
            v = v - np.vdot(b, v) * b
        norm = np.linalg.norm(v)
        if norm > atol:
            basis.append(v / norm)
    return basis


def deflated_matrix(hamiltonian_matrix: np.ndarray, previous_states: Sequence[np.ndarray], betas: Optional[Sequence[float]] = None) -> np.ndarray:
    mat = np.asarray(hamiltonian_matrix, dtype=np.complex128).copy()
    if betas is None:
        betas = [1.0] * len(previous_states)
    for beta, state in zip(betas, previous_states):
        state = np.asarray(state, dtype=np.complex128)
        mat = mat + float(beta) * np.outer(state, state.conj())
    return mat
```

## ansatz/__init__.py

```python
from .base import Ansatz
from .uccsd import UCCSD
from .ryrz import RyRzAnsatz
from .compact import PairExcitationAnsatz, KUpCCGAnsatz, CompactAnsatz, kUPCCGAnsatz
from .adapt import ADAPTAnsatz, AdaptOperator, build_uccsd_operator_pool, build_pair_operator_pool

__all__ = [
    "Ansatz",
    "UCCSD",
    "RyRzAnsatz",
    "PairExcitationAnsatz",
    "KUpCCGAnsatz",
    "CompactAnsatz",
    "kUPCCGAnsatz",
    "ADAPTAnsatz",
    "AdaptOperator",
    "build_uccsd_operator_pool",
    "build_pair_operator_pool",
]
```

## ansatz/adapt.py

```python
"""ADAPT-VQE style adaptive ansatz tools.

ADAPT is not a single fixed circuit.  It is a workflow that grows a circuit by
screening an operator pool and appending the operator with the largest energy
gradient.  The class below stores the selected operators and emits a circuit with
one trainable parameter per selected generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ansatz.base import Ansatz
from backends.core import QuantumCircuit

PauliTerm = Tuple[Tuple[Tuple[int, str], ...], float]
Operator = List[PauliTerm]


def _as_tensor(params):
    if isinstance(params, torch.Tensor):
        return params
    return torch.tensor(params, dtype=torch.float64)


@dataclass
class AdaptOperator:
    """A selected anti-Hermitian generator represented by Pauli terms."""

    label: str
    terms: Operator
    source: Optional[Tuple[int, ...]] = None
    target: Optional[Tuple[int, ...]] = None


class ADAPTAnsatz(Ansatz):
    """Adaptive ansatz that stores selected ADAPT operators.

    Parameters are ordered exactly as operators were added.  The ansatz prepares a
    Hartree-Fock reference state and then applies each generator with a first-order
    product formula.  The Pauli evolution routine mirrors the implementation used
    by UCCSD so that both circuit families compile to the same gate vocabulary.
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int = 0,
        selected_operators: Optional[Sequence[AdaptOperator]] = None,
        trotter_steps: int = 1,
        name: str = "ADAPTAnsatz",
    ):
        self.n_electrons = int(n_electrons)
        self.selected_operators: List[AdaptOperator] = list(selected_operators or [])
        self.trotter_steps = int(trotter_steps)
        self.name = name
        super().__init__(n_qubits, len(self.selected_operators))

    def add_operator(self, operator: AdaptOperator):
        self.selected_operators.append(operator)
        self.n_params = len(self.selected_operators)
        return self

    def extend(self, operators: Sequence[AdaptOperator]):
        for op in operators:
            self.add_operator(op)
        return self

    def copy(self) -> "ADAPTAnsatz":
        return ADAPTAnsatz(
            self.n_qubits,
            self.n_electrons,
            selected_operators=list(self.selected_operators),
            trotter_steps=self.trotter_steps,
            name=self.name,
        )

    def _prepare_hf(self, qc: QuantumCircuit):
        for q in range(min(self.n_electrons, self.n_qubits)):
            qc.x(q)

    @staticmethod
    def _append_pauli_evolution(qc: QuantumCircuit, pauli_string: Tuple[Tuple[int, str], ...], angle):
        if not pauli_string:
            return
        qubits = [int(idx) for idx, _ in pauli_string]
        ops = [str(op).upper() for _, op in pauli_string]
        for q, op in zip(qubits, ops):
            if op == "X":
                qc.h(q)
            elif op == "Y":
                qc.rx(q, np.pi / 2)
            elif op == "Z":
                pass
            else:
                raise ValueError(f"Unsupported Pauli operator '{op}'")
        for left, right in zip(qubits[:-1], qubits[1:]):
            qc.cx(left, right)
        qc.rz(qubits[-1], 2.0 * angle)
        for left, right in reversed(list(zip(qubits[:-1], qubits[1:]))):
            qc.cx(left, right)
        for q, op in reversed(list(zip(qubits, ops))):
            if op == "X":
                qc.h(q)
            elif op == "Y":
                qc.rx(q, -np.pi / 2)

    def forward(self, params) -> QuantumCircuit:
        params = _as_tensor(params)
        if params.numel() != self.n_params:
            raise ValueError(f"ADAPTAnsatz expected {self.n_params} parameters, got {params.numel()}")
        qc = QuantumCircuit(self.n_qubits, name=self.name)
        self._prepare_hf(qc)
        for _ in range(self.trotter_steps):
            for theta, op in zip(params, self.selected_operators):
                for pauli_string, coeff in op.terms:
                    self._append_pauli_evolution(qc, pauli_string, theta * coeff)
        return qc

    def zero_parameters(self) -> torch.Tensor:
        return torch.zeros(self.n_params, dtype=torch.float64)


def _jw_single_excitation(i: int, a: int) -> Operator:
    """Return a compact JW-like anti-Hermitian single excitation generator.

    The exact Jordan-Wigner image of a_i^dagger a_j - h.c. contains X/Y strings
    and parity Z strings.  This helper creates the same physically meaningful
    rotation pattern without requiring OpenFermion at import time.
    """

    lo, hi = sorted((i, a))
    z_string = tuple((q, "Z") for q in range(lo + 1, hi))
    return [
        (tuple([(lo, "X"), *z_string, (hi, "Y")]), 0.5),
        (tuple([(lo, "Y"), *z_string, (hi, "X")]), -0.5),
    ]


def _jw_pair_excitation(i: int, j: int, a: int, b: int) -> Operator:
    """Return an approximate spin-adapted double-excitation Pauli generator."""

    qubits = [i, j, a, b]
    patterns = [
        ("XXXY", 1.0 / 8.0),
        ("XXYX", 1.0 / 8.0),
        ("XYXX", -1.0 / 8.0),
        ("YXXX", -1.0 / 8.0),
        ("YYXY", 1.0 / 8.0),
        ("YYYX", 1.0 / 8.0),
        ("YXYX", -1.0 / 8.0),
        ("XYYY", -1.0 / 8.0),
    ]
    out: Operator = []
    for pattern, coeff in patterns:
        out.append((tuple((q, p) for q, p in zip(qubits, pattern)), coeff))
    return out


def build_uccsd_operator_pool(n_qubits: int, n_electrons: int, include_singles: bool = True, include_doubles: bool = True) -> List[AdaptOperator]:
    """Build a singles/doubles pool for ADAPT screening."""

    occ = list(range(n_electrons))
    vir = list(range(n_electrons, n_qubits))
    pool: List[AdaptOperator] = []
    if include_singles:
        for i in occ:
            for a in vir:
                pool.append(AdaptOperator(label=f"S({i}->{a})", terms=_jw_single_excitation(i, a), source=(i,), target=(a,)))
    if include_doubles:
        for x, i in enumerate(occ):
            for j in occ[x + 1 :]:
                for y, a in enumerate(vir):
                    for b in vir[y + 1 :]:
                        pool.append(
                            AdaptOperator(
                                label=f"D({i},{j}->{a},{b})",
                                terms=_jw_pair_excitation(i, j, a, b),
                                source=(i, j),
                                target=(a, b),
                            )
                        )
    return pool


def build_pair_operator_pool(n_qubits: int, n_electrons: int) -> List[AdaptOperator]:
    """Build a compact pair-only ADAPT pool."""

    if n_qubits % 2 != 0 or n_electrons % 2 != 0:
        raise ValueError("Pair pool requires even n_qubits and n_electrons")
    n_occ_pairs = n_electrons // 2
    n_pairs = n_qubits // 2
    pool: List[AdaptOperator] = []
    for p in range(n_occ_pairs):
        for q in range(n_occ_pairs, n_pairs):
            i, j = 2 * p, 2 * p + 1
            a, b = 2 * q, 2 * q + 1
            pool.append(AdaptOperator(label=f"P({p}->{q})", terms=_jw_pair_excitation(i, j, a, b), source=(i, j), target=(a, b)))
    return pool


def finite_difference_gradient(
    energy_fn: Callable[[torch.Tensor], float],
    params: torch.Tensor,
    operator_index: int,
    epsilon: float = 1e-4,
) -> float:
    """Estimate the gradient for a newly appended ADAPT operator.

    This helper is intentionally optimizer-agnostic.  A caller can append a trial
    operator with zero parameter, evaluate plus/minus shifts, then choose the
    largest absolute gradient.
    """

    shift = torch.zeros_like(params)
    shift[operator_index] = epsilon
    e_plus = float(energy_fn(params + shift))
    e_minus = float(energy_fn(params - shift))
    return (e_plus - e_minus) / (2.0 * epsilon)


def select_largest_gradient(gradients: Sequence[float], pool: Sequence[AdaptOperator], threshold: float = 1e-4):
    if len(gradients) != len(pool):
        raise ValueError("Gradient list and operator pool must have the same length")
    if not gradients:
        return None, 0.0, True
    idx = int(np.argmax(np.abs(np.asarray(gradients, dtype=float))))
    max_grad = float(gradients[idx])
    converged = abs(max_grad) < threshold
    return pool[idx], max_grad, converged
```

## ansatz/compact.py

```python
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
```

## ansatz/ryrz.py

```python
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
```

## backends/Cirq.py

```python
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
```

## backends/Qulacs.py

```python
"""Qulacs backend adapter for HYQ-ALG-LIB.

Qulacs is a high-performance C++/Python simulator for variational quantum
circuits.  This adapter provides a native state-vector path when Qulacs is
installed and uses the shared dense simulator for sampling so that the return
format matches the existing backend contract.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import QuantumBackend
from .core import QuantumCircuit
from ._matrix_simulator import gate_matrix, sample_counts, simulate_statevector, statevector_to_torch, _to_float


class QulacsBackend(QuantumBackend):
    """HYQ backend powered by Qulacs.

    The native translator uses Qulacs dense matrices for maximum compatibility
    with the internal HYQ gate set.  Qulacs also has optimized built-in gates;
    dense matrices are chosen here because the project already defines a broad
    collection of gates and control modifiers in one unified format.
    """

    def __init__(self, seed: Optional[int] = None, use_native: bool = True):
        self.seed = seed
        self.use_native = use_native
        self._qulacs = None

    def _check_qulacs(self):
        if self._qulacs is not None:
            return self._qulacs
        try:
            import qulacs
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("QulacsBackend requires 'qulacs'. Install it with: pip install qulacs") from exc
        self._qulacs = qulacs
        return qulacs

    def _dense_gate(self, inst):
        from qulacs.gate import DenseMatrix

        if getattr(inst, "matrix", None) is not None:
            mat = np.asarray(inst.matrix, dtype=np.complex128)
        else:
            mat = gate_matrix(getattr(inst, "name", ""), getattr(inst, "params", []))
        if mat is None:
            raise NotImplementedError(f"QulacsBackend does not support gate '{getattr(inst, 'name', None)}'")
        targets = list(getattr(inst, "qubits", []))
        gate = DenseMatrix(targets, mat)
        for q, v in zip(getattr(inst, "control_qubits", []) or [], getattr(inst, "control_values", []) or []):
            gate.add_control_qubit(int(q), int(v))
        return gate

    def _to_qulacs_circuit(self, circuit: QuantumCircuit):
        qulacs = self._check_qulacs()
        qc = qulacs.QuantumCircuit(circuit.n_qubits)
        for inst in circuit.instructions:
            if getattr(inst, "circuit", None) is not None:
                # Turn a sub-circuit into a dense matrix by simulation of basis states.
                sub = inst.circuit
                dim = 2**sub.n_qubits
                mat = np.zeros((dim, dim), dtype=np.complex128)
                for j in range(dim):
                    basis = np.zeros(dim, dtype=np.complex128)
                    basis[j] = 1.0
                    mat[:, j] = simulate_statevector(sub, basis)
                from qulacs.gate import DenseMatrix

                gate = DenseMatrix(list(inst.qubits), mat)
                qc.add_gate(gate)
                continue
            if getattr(inst, "name", "") == "GlobalPhase":
                theta = _to_float(inst.params[0]) if inst.params else 0.0
                mat = np.exp(1.0j * theta) * np.eye(1, dtype=np.complex128)
                # Global phase has no direct target; skip because it does not affect
                # measurement and only shifts the global state phase.
                continue
            qc.add_gate(self._dense_gate(inst))
        return qc

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        # Qulacs supports sampling, but the dense utility keeps bit ordering
        # consistent with QiskitBackend and the rest of the library.
        state = simulate_statevector(circuit)
        return sample_counts(state, shots, measure_qubits, seed=self.seed)

    def get_statevector(self, circuit) -> torch.Tensor:
        if not self.use_native:
            return statevector_to_torch(simulate_statevector(circuit))
        try:
            qulacs = self._check_qulacs()
            state = qulacs.QuantumState(circuit.n_qubits)
            state.set_zero_state()
            qc = self._to_qulacs_circuit(circuit)
            qc.update_quantum_state(state)
            vector = np.asarray(state.get_vector(), dtype=np.complex128)
            return torch.tensor(vector, dtype=torch.complex128)
        except Exception:
            return statevector_to_torch(simulate_statevector(circuit))
```

## backends/Qutip.py

```python
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
```

## backends/__init__.py

```python
"""Extended backend exports for HYQ-ALG-LIB."""

try:
    from .Qiskit import QiskitBackend
except Exception:  # pragma: no cover
    QiskitBackend = None  # type: ignore
try:
    from .Pennylane import PennylaneBackend
except Exception:  # pragma: no cover
    PennylaneBackend = None  # type: ignore
try:
    from .Tensorcircuit import TensorCircuitBackend
except Exception:  # pragma: no cover
    TensorCircuitBackend = None  # type: ignore

from .Cirq import CirqBackend
from .Qulacs import QulacsBackend
from .Qutip import QutipBackend

EXTENDED_BACKENDS = {
    "cirq": CirqBackend,
    "qulacs": QulacsBackend,
    "qutip": QutipBackend,
}

__all__ = [
    "CirqBackend",
    "QulacsBackend",
    "QutipBackend",
    "EXTENDED_BACKENDS",
]
```

## backends/_matrix_simulator.py

```python
"""
Dense state-vector utilities shared by optional HYQ-ALG-LIB backends.

The original project stores circuits in a lightweight backend-independent
``QuantumCircuit`` object.  Each instruction contains a gate name, target qubits,
optional parameters, optional control qubits, and optional dense matrix.  The
helpers in this file translate that representation into a pure NumPy state-vector
simulation.  Optional backends such as Cirq, Qulacs, and QuTiP can use these
functions as a correctness reference and as a fallback when a native gate is not
available.

The code intentionally avoids importing Qiskit/PennyLane/TensorCircuit.  It is
therefore safe to import even when only NumPy and Torch are installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # torch is already used by the original backend API, but keep import optional.
    import torch
except Exception:  # pragma: no cover - import guard for documentation builds
    torch = None  # type: ignore


_EPS = 1e-12


def _to_float(value) -> float:
    """Convert Python/NumPy/Torch scalar-like values to a plain float."""

    if torch is not None and hasattr(value, "detach"):
        return float(value.detach().cpu().numpy())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _complex_array(matrix) -> np.ndarray:
    """Return a complex128 NumPy array without modifying the input object."""

    if torch is not None and hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.complex128)


def _identity() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def _x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _y() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)


def _z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _h() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


def _s() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)


def _sdg() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)


def _t() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)


def _tdg() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(-1.0j * np.pi / 4.0)]], dtype=np.complex128)


def _sx() -> np.ndarray:
    return 0.5 * np.array(
        [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=np.complex128
    )


def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=np.complex128
    )


def _phase(theta: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * theta)]], dtype=np.complex128)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array(
        [
            [c, -np.exp(1.0j * lam) * s],
            [np.exp(1.0j * phi) * s, np.exp(1.0j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def _swap() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )


def _iswap() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )


def _rxx(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * np.eye(4, dtype=np.complex128) - 1j * math.sin(theta / 2.0) * np.kron(_x(), _x())


def _ryy(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * np.eye(4, dtype=np.complex128) - 1j * math.sin(theta / 2.0) * np.kron(_y(), _y())


def _rzz(theta: float) -> np.ndarray:
    return np.diag(
        [np.exp(-0.5j * theta), np.exp(0.5j * theta), np.exp(0.5j * theta), np.exp(-0.5j * theta)]
    ).astype(np.complex128)


def _controlled(base: np.ndarray, ctrl_value: int = 1) -> np.ndarray:
    """Return a one-control version of a k-qubit base unitary."""

    dim = base.shape[0]
    result = np.eye(2 * dim, dtype=np.complex128)
    start = dim if ctrl_value == 1 else 0
    result[start : start + dim, start : start + dim] = base
    return result


def gate_matrix(name: str, params: Optional[Sequence] = None) -> Optional[np.ndarray]:
    """Return a dense unitary matrix for a HYQ gate name.

    Unsupported names return ``None`` so that native backends can choose their own
    implementation or skip/raise a useful error.
    """

    params = list(params or [])
    key = name.lower()
    single_no_param = {
        "id": _identity,
        "i": _identity,
        "x": _x,
        "y": _y,
        "z": _z,
        "h": _h,
        "s": _s,
        "sdg": _sdg,
        "t": _t,
        "tdg": _tdg,
        "sx": _sx,
    }
    if key in single_no_param:
        return single_no_param[key]()
    if key in {"rx"}:
        return _rx(_to_float(params[0]))
    if key in {"ry"}:
        return _ry(_to_float(params[0]))
    if key in {"rz"}:
        return _rz(_to_float(params[0]))
    if key in {"p", "phase"}:
        return _phase(_to_float(params[0]))
    if key in {"u", "u3"}:
        return _u3(_to_float(params[0]), _to_float(params[1]), _to_float(params[2]))
    if key in {"swap"}:
        return _swap()
    if key in {"iswap"}:
        return _iswap()
    if key in {"rxx"}:
        return _rxx(_to_float(params[0]))
    if key in {"ryy"}:
        return _ryy(_to_float(params[0]))
    if key in {"rzz"}:
        return _rzz(_to_float(params[0]))
    if key in {"cnot", "cx"}:
        return _controlled(_x(), 1)
    if key == "cy":
        return _controlled(_y(), 1)
    if key == "cz":
        return _controlled(_z(), 1)
    if key == "ch":
        return _controlled(_h(), 1)
    if key in {"cp", "cphase"}:
        return _controlled(_phase(_to_float(params[0])), 1)
    if key == "crx":
        return _controlled(_rx(_to_float(params[0])), 1)
    if key == "cry":
        return _controlled(_ry(_to_float(params[0])), 1)
    if key == "crz":
        return _controlled(_rz(_to_float(params[0])), 1)
    if key == "ccx":
        mat = np.eye(8, dtype=np.complex128)
        mat[6, 6] = 0
        mat[7, 7] = 0
        mat[6, 7] = 1
        mat[7, 6] = 1
        return mat
    if key == "cswap":
        mat = np.eye(8, dtype=np.complex128)
        mat[5, 5] = 0
        mat[6, 6] = 0
        mat[5, 6] = 1
        mat[6, 5] = 1
        return mat
    return None


def zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


def _reorder_operator_for_statevector(operator: np.ndarray, qubits: Sequence[int], n_qubits: int) -> np.ndarray:
    """Embed a k-qubit operator into an n-qubit Hilbert space.

    Qubit 0 is treated as the most significant bit, matching the bitstring format
    used by the rest of this module.  The implementation prioritizes clarity over
    speed; the optional native backends provide fast paths for large circuits.
    """

    qubits = list(qubits)
    k = len(qubits)
    if k == 0:
        return np.eye(2**n_qubits, dtype=np.complex128)
    if operator.shape != (2**k, 2**k):
        raise ValueError(f"Operator shape {operator.shape} does not match {k} target qubits")
    dim = 2**n_qubits
    full = np.zeros((dim, dim), dtype=np.complex128)
    qubit_set = set(qubits)
    spectator_qubits = [q for q in range(n_qubits) if q not in qubit_set]

    for col in range(dim):
        col_bits = [(col >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        local_col = 0
        for q in qubits:
            local_col = (local_col << 1) | col_bits[q]
        spectator_pattern = tuple(col_bits[q] for q in spectator_qubits)
        for local_row in range(2**k):
            amp = operator[local_row, local_col]
            if abs(amp) < _EPS:
                continue
            row_bits = list(col_bits)
            for offset, q in enumerate(reversed(qubits)):
                # local_row is encoded with qubits[0] as most significant bit.
                bit = (local_row >> offset) & 1
                row_bits[q] = bit
            if tuple(row_bits[q] for q in spectator_qubits) != spectator_pattern:
                continue
            row = 0
            for bit in row_bits:
                row = (row << 1) | bit
            full[row, col] += amp
    return full


def apply_operator(state: np.ndarray, operator: np.ndarray, qubits: Sequence[int], n_qubits: int) -> np.ndarray:
    full = _reorder_operator_for_statevector(operator, qubits, n_qubits)
    return full @ state


def _controlled_full_matrix(base: np.ndarray, target_qubits: Sequence[int], control_qubits: Sequence[int], control_values: Sequence[int], n_qubits: int) -> np.ndarray:
    """Create a full-system matrix for a controlled operation."""

    dim = 2**n_qubits
    out = np.eye(dim, dtype=np.complex128)
    target_qubits = list(target_qubits)
    control_qubits = list(control_qubits)
    control_values = list(control_values)
    qubit_set = set(target_qubits)
    for q in control_qubits:
        if q in qubit_set:
            raise ValueError("Control qubits must not overlap target qubits")

    k = len(target_qubits)
    for col in range(dim):
        col_bits = [(col >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if any(col_bits[q] != v for q, v in zip(control_qubits, control_values)):
            continue
        local_col = 0
        for q in target_qubits:
            local_col = (local_col << 1) | col_bits[q]
        # Remove identity rows for this column; controlled block replaces them.
        out[col, col] = 0.0
        for local_row in range(2**k):
            amp = base[local_row, local_col]
            if abs(amp) < _EPS:
                continue
            row_bits = list(col_bits)
            for offset, q in enumerate(reversed(target_qubits)):
                row_bits[q] = (local_row >> offset) & 1
            row = 0
            for bit in row_bits:
                row = (row << 1) | bit
            out[row, col] += amp
    return out


def instruction_matrix(inst, n_qubits: int) -> Tuple[np.ndarray, List[int], bool]:
    """Return ``(matrix, qubits, already_full_system)`` for a HYQ instruction."""

    if getattr(inst, "matrix", None) is not None:
        mat = _complex_array(inst.matrix)
        targets = list(getattr(inst, "qubits", []))
    else:
        mat = gate_matrix(getattr(inst, "name", ""), getattr(inst, "params", []))
        targets = list(getattr(inst, "qubits", []))
    if mat is None:
        raise NotImplementedError(f"Unsupported gate in dense simulator: {getattr(inst, 'name', None)}")

    control_qubits = list(getattr(inst, "control_qubits", []) or [])
    control_values = list(getattr(inst, "control_values", []) or [])
    if control_qubits:
        full = _controlled_full_matrix(mat, targets, control_qubits, control_values, n_qubits)
        return full, list(range(n_qubits)), True
    return mat, targets, False


def simulate_statevector(circuit, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulate a HYQ ``QuantumCircuit`` and return a NumPy state vector."""

    n_qubits = int(circuit.n_qubits)
    state = zero_state(n_qubits) if initial_state is None else _complex_array(initial_state).copy()
    if state.shape != (2**n_qubits,):
        raise ValueError(f"Initial state shape {state.shape} incompatible with {n_qubits} qubits")
    for inst in circuit.instructions:
        if getattr(inst, "circuit", None) is not None:
            # Simulate sub-circuit and apply it as a dense unitary by columns.
            sub = inst.circuit
            sub_dim = 2**sub.n_qubits
            unitary = np.zeros((sub_dim, sub_dim), dtype=np.complex128)
            for j in range(sub_dim):
                basis = np.zeros(sub_dim, dtype=np.complex128)
                basis[j] = 1.0
                unitary[:, j] = simulate_statevector(sub, basis)
            state = apply_operator(state, unitary, list(inst.qubits), n_qubits)
            continue
        name = getattr(inst, "name", "")
        if name == "GlobalPhase":
            theta = _to_float(inst.params[0]) if inst.params else 0.0
            state = np.exp(1.0j * theta) * state
            continue
        mat, qubits, already_full = instruction_matrix(inst, n_qubits)
        if already_full:
            state = mat @ state
        else:
            state = apply_operator(state, mat, qubits, n_qubits)
    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm
    return state


def statevector_to_torch(state: np.ndarray):
    """Convert a NumPy complex vector to the Torch type expected by HYQ backends."""

    if torch is None:
        return state
    return torch.tensor(state, dtype=torch.complex128)


def probability_distribution(state: np.ndarray) -> np.ndarray:
    probs = np.abs(state) ** 2
    total = probs.sum()
    if total <= 0:
        raise ValueError("State vector has zero norm")
    return probs / total


def bitstring(index: int, n_qubits: int, selected: Optional[Sequence[int]] = None) -> str:
    bits = [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
    if selected is None:
        return "".join(str(b) for b in bits)
    return "".join(str(bits[q]) for q in selected)


def sample_counts(state: np.ndarray, shots: int, measure_qubits: Sequence[int], seed: Optional[int] = None) -> Dict[str, int]:
    """Sample computational-basis bitstrings from a state vector."""

    n_qubits = int(round(math.log2(state.size)))
    probs = probability_distribution(state)
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(state.size, size=int(shots), p=probs)
    counts: Dict[str, int] = {}
    for outcome in outcomes:
        key = bitstring(int(outcome), n_qubits, measure_qubits)
        counts[key] = counts.get(key, 0) + 1
    return counts


def expectation_from_pauli_terms(state: np.ndarray, terms: Mapping[Tuple[Tuple[int, str], ...], complex], n_qubits: int) -> complex:
    """Compute <psi|H|psi> for an OpenFermion-like Pauli dictionary."""

    value = 0.0 + 0.0j
    for term, coeff in terms.items():
        if not term:
            value += coeff
            continue
        ops = []
        qubits = []
        for q, pauli in term:
            qubits.append(int(q))
            p = str(pauli).upper()
            if p == "X":
                ops.append(_x())
            elif p == "Y":
                ops.append(_y())
            elif p == "Z":
                ops.append(_z())
            else:
                ops.append(_identity())
        local = ops[0]
        for op in ops[1:]:
            local = np.kron(local, op)
        transformed = apply_operator(state, local, qubits, n_qubits)
        value += coeff * np.vdot(state, transformed)
    return value


@dataclass
class StatevectorResult:
    """Small container used by examples and algorithm modules."""

    state: np.ndarray
    n_qubits: int

    @property
    def probabilities(self) -> np.ndarray:
        return probability_distribution(self.state)

    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None, seed: Optional[int] = None) -> Dict[str, int]:
        return sample_counts(self.state, shots, list(range(self.n_qubits)) if qubits is None else qubits, seed=seed)
```

## backends/extended_registry.py

```python
"""Small backend registry for projects that want dynamic backend selection."""

from __future__ import annotations

import os
from typing import Any, Dict, Type

from .Cirq import CirqBackend
from .Qulacs import QulacsBackend
from .Qutip import QutipBackend

try:
    from .Qiskit import QiskitBackend
except Exception:  # pragma: no cover
    QiskitBackend = None  # type: ignore
try:
    from .Pennylane import PennylaneBackend
except Exception:  # pragma: no cover
    PennylaneBackend = None  # type: ignore
try:
    from .Tensorcircuit import TensorCircuitBackend
except Exception:  # pragma: no cover
    TensorCircuitBackend = None  # type: ignore

_BACKEND_CLASSES: Dict[str, Type] = {
    "cirq": CirqBackend,
    "qulacs": QulacsBackend,
    "qutip": QutipBackend,
}
if QiskitBackend is not None:
    _BACKEND_CLASSES["qiskit"] = QiskitBackend
if PennylaneBackend is not None:
    _BACKEND_CLASSES["pennylane"] = PennylaneBackend
if TensorCircuitBackend is not None:
    _BACKEND_CLASSES["tensorcircuit"] = TensorCircuitBackend

_ACTIVE_BACKEND = None


def available_backends():
    return sorted(_BACKEND_CLASSES.keys())


def register_backend(name: str, backend_cls: Type):
    _BACKEND_CLASSES[name.lower()] = backend_cls


def set_backend(name: str, **kwargs):
    global _ACTIVE_BACKEND
    key = name.lower()
    if key not in _BACKEND_CLASSES:
        raise ValueError(f"Unknown backend '{name}'. Available: {available_backends()}")
    _ACTIVE_BACKEND = _BACKEND_CLASSES[key](**kwargs)
    os.environ["HYQ_BACKEND"] = key
    return _ACTIVE_BACKEND


def get_backend(default: str = "qiskit", **kwargs):
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        name = os.environ.get("HYQ_BACKEND", default)
        return set_backend(name, **kwargs)
    return _ACTIVE_BACKEND
```

## requirements-extension.txt

```
# Optional dependencies for the extension modules.
# Install only the backends you need.
cirq>=1.3
qulacs>=0.6
qutip>=5.0
qutip-qip>=0.3
scipy>=1.10
numpy>=1.23
torch>=2.0
```

