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
