"""Variational Quantum Deflation helpers for excited states.

The helpers in this module are backend-independent NumPy utilities.  They are
intended to support VQD-style excited-state workflows by adding overlap penalties
against previously found states,

    E_VQD(theta) = <psi(theta)|H|psi(theta)> + sum_i beta_i |<psi(theta)|phi_i>|^2,

or equivalently, for exact classical checks,

    H_deflated = H + sum_i beta_i |phi_i><phi_i|.

All state vectors are normalized before overlaps/projectors are evaluated by
default, which makes the helpers robust to simulator outputs or user-supplied
vectors that are not already unit norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class VQDObjectiveBreakdown:
    energy: float
    penalties: List[float]
    total: float
    overlaps: Optional[List[float]] = None


def _as_state_vector(state: np.ndarray, name: str = "state") -> np.ndarray:
    vec = np.asarray(state, dtype=np.complex128)
    if vec.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional state vector, got shape {vec.shape}")
    if vec.size == 0:
        raise ValueError(f"{name} must not be empty")
    return vec


def _normalized_state(state: np.ndarray, name: str = "state") -> np.ndarray:
    vec = _as_state_vector(state, name)
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError(f"{name} has zero norm")
    return vec / norm


def _validate_betas(previous_states: Sequence[np.ndarray], betas: Optional[Sequence[float]]) -> List[float]:
    n_prev = len(previous_states)
    if betas is None:
        return [1.0] * n_prev
    beta_list = [float(beta) for beta in betas]
    if len(beta_list) != n_prev:
        raise ValueError(f"Expected {n_prev} beta values for {n_prev} previous states, got {len(beta_list)}")
    if any((not np.isfinite(beta)) or beta < 0.0 for beta in beta_list):
        raise ValueError("VQD deflation beta values must be finite and non-negative")
    return beta_list


def state_overlap(state_a: np.ndarray, state_b: np.ndarray, normalize: bool = True) -> float:
    """Return |<state_a|state_b>|^2.

    By default the input states are normalized first.  This matches the physical
    definition of state overlap and prevents non-unit simulator/user vectors from
    producing overlaps larger than one.
    """

    if normalize:
        a = _normalized_state(state_a, "state_a")
        b = _normalized_state(state_b, "state_b")
    else:
        a = _as_state_vector(state_a, "state_a")
        b = _as_state_vector(state_b, "state_b")
    if a.shape != b.shape:
        raise ValueError(f"State vectors must have the same shape, got {a.shape} and {b.shape}")
    overlap = abs(np.vdot(a, b)) ** 2
    return float(np.clip(overlap.real, 0.0, 1.0) if normalize else overlap.real)


def state_energy(hamiltonian_matrix: np.ndarray, state: np.ndarray, normalize: bool = True) -> float:
    """Return the Rayleigh quotient <state|H|state>/<state|state>."""

    mat = np.asarray(hamiltonian_matrix, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"hamiltonian_matrix must be square, got shape {mat.shape}")
    vec = _normalized_state(state) if normalize else _as_state_vector(state)
    if mat.shape[0] != vec.size:
        raise ValueError(f"Matrix dimension {mat.shape[0]} does not match state size {vec.size}")
    value = np.vdot(vec, mat @ vec)
    if abs(value.imag) > 1e-8:
        raise ValueError(f"Energy expectation has a non-negligible imaginary part: {value}")
    return float(value.real)


def vqd_objective(
    energy: float,
    candidate_state: np.ndarray,
    previous_states: Sequence[np.ndarray],
    betas: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> VQDObjectiveBreakdown:
    """Evaluate the VQD objective from a precomputed candidate energy."""

    beta_list = _validate_betas(previous_states, betas)
    overlaps = [state_overlap(candidate_state, state, normalize=normalize) for state in previous_states]
    penalties = [float(beta) * overlap for beta, overlap in zip(beta_list, overlaps)]
    total = float(energy + sum(penalties))
    return VQDObjectiveBreakdown(float(energy), penalties, total, overlaps)


def vqd_objective_from_matrix(
    hamiltonian_matrix: np.ndarray,
    candidate_state: np.ndarray,
    previous_states: Sequence[np.ndarray],
    betas: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> VQDObjectiveBreakdown:
    """Evaluate the full VQD objective directly from a Hamiltonian matrix."""

    energy = state_energy(hamiltonian_matrix, candidate_state, normalize=normalize)
    return vqd_objective(
        energy=energy,
        candidate_state=candidate_state,
        previous_states=previous_states,
        betas=betas,
        normalize=normalize,
    )


def gram_schmidt_states(states: Sequence[np.ndarray], atol: float = 1e-12) -> List[np.ndarray]:
    """Return an orthonormal basis spanning the nonzero input states."""

    if atol < 0:
        raise ValueError("atol must be non-negative")
    basis: List[np.ndarray] = []
    expected_size: Optional[int] = None
    for idx, state in enumerate(states):
        v = _as_state_vector(state, f"states[{idx}]").copy()
        if expected_size is None:
            expected_size = v.size
        elif v.size != expected_size:
            raise ValueError(f"All states must have the same size; got {v.size} and expected {expected_size}")
        for b in basis:
            v = v - np.vdot(b, v) * b
        norm = np.linalg.norm(v)
        if norm > atol:
            basis.append(v / norm)
    return basis


def projector(state: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Return |state><state|, normalizing the state by default."""

    vec = _normalized_state(state) if normalize else _as_state_vector(state)
    return np.outer(vec, vec.conj())


def deflated_matrix(
    hamiltonian_matrix: np.ndarray,
    previous_states: Sequence[np.ndarray],
    betas: Optional[Sequence[float]] = None,
    normalize: bool = True,
    symmetrize: bool = True,
) -> np.ndarray:
    """Return H + sum_i beta_i |phi_i><phi_i| for VQD deflation."""

    mat = np.asarray(hamiltonian_matrix, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"hamiltonian_matrix must be square, got shape {mat.shape}")

    out = mat.copy()
    beta_list = _validate_betas(previous_states, betas)
    for idx, (beta, state) in enumerate(zip(beta_list, previous_states)):
        vec = _normalized_state(state, f"previous_states[{idx}]") if normalize else _as_state_vector(state, f"previous_states[{idx}]")
        if vec.size != out.shape[0]:
            raise ValueError(f"previous_states[{idx}] size {vec.size} does not match matrix dimension {out.shape[0]}")
        out = out + float(beta) * np.outer(vec, vec.conj())

    if symmetrize:
        out = 0.5 * (out + out.conj().T)
    return out


def deflated_eigensystem(
    hamiltonian_matrix: np.ndarray,
    previous_states: Sequence[np.ndarray],
    betas: Optional[Sequence[float]] = None,
    normalize: bool = True,
):
    """Diagonalize the deflated matrix, useful as a classical VQD reference."""

    mat = deflated_matrix(
        hamiltonian_matrix,
        previous_states=previous_states,
        betas=betas,
        normalize=normalize,
    )
    vals, vecs = np.linalg.eigh(mat)
    return vals, vecs
