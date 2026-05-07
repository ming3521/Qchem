import torch
import numpy as np
import time
from .base import QuantumSolver

class SSQITESolver(QuantumSolver):
    """
    SSQITE (Subspace-Search QITE)
    """
    def solve(self, ansatz_func, init_params, hamiltonian, n_states=2, b_step=0.2, max_steps=100, regularization=1e-4):
        params = init_params.clone().detach().to(dtype=torch.float64)
        params.requires_grad_(True)
        n_params = len(params)
        
        temp_qc = ansatz_func(params)
        n_qubits = temp_qc.n_qubits
        H_matrix = self._pauli_string_to_matrix(n_qubits, hamiltonian)
        
        d_tau = [b_step / (2**k) for k in range(n_states)]
        
        history_energies = [] 
        print(f"--- SSQITE Start: States={n_states}, b={b_step}, Steps={max_steps} ---")
        print(f"--- Weights (Time Steps): {d_tau} ---")
        
        for step in range(max_steps):
            t0 = time.time()
            total_delta_theta = torch.zeros(n_params, dtype=torch.float64)
            current_energies = []
            
            for k in range(n_states):
                A_k, C_k, E_k = self._compute_AC_single_state(
                    ansatz_func, params, H_matrix, n_qubits, state_idx=k
                )
                current_energies.append(E_k)
                
                A_np = A_k.detach().numpy() + regularization * np.eye(n_params)
                C_np = C_k.detach().numpy()
                
                try:
                    dot_theta_k, _, _, _ = np.linalg.lstsq(A_np, C_np, rcond=1e-5)
                except Exception:
                    dot_theta_k = np.zeros(n_params)
                
                total_delta_theta += d_tau[k] * torch.tensor(dot_theta_k)

            history_energies.append(current_energies)
            
            with torch.no_grad():
                new_params_data = params + total_delta_theta
            
            params = new_params_data.clone().detach().requires_grad_(True)
            
            t1 = time.time()
            if step % 10 == 0 or step == max_steps - 1:
                e_str = ", ".join([f"{e:.4f}" for e in current_energies])
                print(f"Step {step:<3} | Energies: [{e_str}] ({t1-t0:.3f}s)")
                
        return history_energies[-1], params, history_energies

    def _compute_AC_single_state(self, ansatz_func, params, H_matrix, n_qubits, state_idx):
        def get_psi_k(p):
            qc = ansatz_func(p)
            self._prepend_state_preparation(qc, state_idx) 
            return self.backend.get_statevector(qc)

        def get_psi_real(p): return torch.real(get_psi_k(p))
        def get_psi_imag(p): return torch.imag(get_psi_k(p))

        J_real = torch.autograd.functional.jacobian(get_psi_real, params)
        J_imag = torch.autograd.functional.jacobian(get_psi_imag, params)
        J = J_real + 1j * J_imag
        
        A = torch.real(torch.matmul(J.conj().T, J))
        
        with torch.no_grad():
            psi = get_psi_k(params)
            H_psi = torch.matmul(H_matrix, psi)
            E = torch.vdot(psi, H_psi).real.item()
            
        C = -torch.real(torch.matmul(J.conj().T, H_psi))
        
        return A, C, E

    def _prepend_state_preparation(self, qc, state_idx):
        n_qubits = qc.n_qubits
        temp_qc = type(qc)(n_qubits) 
        
        for j in range(n_qubits):
            if (state_idx >> j) & 1:
                target_qubit = n_qubits - 1 - j
                temp_qc.x(target_qubit)
                
        if temp_qc.instructions:
            qc.instructions = temp_qc.instructions + qc.instructions