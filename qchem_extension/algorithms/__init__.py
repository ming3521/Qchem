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
