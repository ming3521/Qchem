import torch
from .base import QuantumSolver

class VQESolver(QuantumSolver):
    """
    VQE (Variational Quantum Eigensolver)
    """
    def solve(self, ansatz_func, init_params, hamiltonian, steps=100, lr=0.1):
        params = init_params.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)
        
        temp_qc = ansatz_func(params)
        n_qubits = temp_qc.n_qubits
        H_matrix = self._pauli_string_to_matrix(n_qubits, hamiltonian)
        
        loss_history = []
        print(f"--- VQE Start (Steps={steps}) ---")
        
        energy = None
        for i in range(steps):
            optimizer.zero_grad()
            qc = ansatz_func(params)
            state = self.backend.get_statevector(qc)
            H_psi = torch.matmul(H_matrix, state)
            energy = torch.vdot(state, H_psi).real
            
            energy.backward()
            optimizer.step()
            loss_history.append(energy.item())
            
            if i % 20 == 0:
                print(f"VQE Step {i}: Energy = {energy.item():.6f}")
                
        print(f"VQE Final Energy: {energy.item():.6f}")
        return energy.item(), params.detach(), loss_history