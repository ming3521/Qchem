import torch
import numpy as np
import time
from scipy.linalg import pinv
from .base import QuantumSolver

class SAQITESolver(QuantumSolver):
    """
    SA-QITE (Stochastic Approximation / SPSA-based VarQITE) 
    """
    def solve(self, ansatz_func, init_params, hamiltonian, total_tau=1.0, delta_tau=0.1, 
                n_samples=50, epsilon=0.1, tau1=0.9, tau2=0.0, delta=0.1): 
        params = init_params.clone().detach().numpy().astype(np.float64)
        n_params = len(params)
        
        temp_qc = ansatz_func(torch.tensor(params))
        n_qubits = temp_qc.n_qubits
        H_matrix = self._pauli_string_to_matrix(n_qubits, hamiltonian)
        
        n_steps = int(total_tau / delta_tau)
        energies = []
        params_history = []
        
        g_hat = np.eye(n_params) # 初始化度量矩阵
        b_hat = np.zeros(n_params)
        
        print(f"--- SA-QITE Start (N={n_samples}, Total Tau={total_tau}) ---")
        
        for step in range(n_steps):
            t0 = time.time()
            
            current_energy = self._compute_energy_exact(ansatz_func, params, H_matrix)
            energies.append(current_energy)
            
            # 1. 估计量子几何张量 (QGT/FIM)
            qgt_samples = [self._compute_spsa_qgt_sample(ansatz_func, params, epsilon) 
                           for _ in range(n_samples)]
            qgt_avg = np.mean(qgt_samples, axis=0)
            
            # 平滑更新 G
            g_hat = tau1 * g_hat + (1 - tau1) * qgt_avg
            # 正则化 G
            g_reg = g_hat + delta * np.eye(n_params)
            
            # 2. 估计梯度
            grad_samples = [self._compute_spsa_grad_sample(ansatz_func, params, H_matrix, epsilon) 
                            for _ in range(n_samples)]
            grad_avg = np.mean(grad_samples, axis=0)
            
            # 平滑更新 b
            b_hat = tau2 * b_hat + (1 - tau2) * grad_avg
            
            # 3. 求解更新方向
            try:
                # 使用 lstsq 替代 pinv 以获得更好稳定性
                dot_theta, _, _, _ = np.linalg.lstsq(g_reg, -b_hat, rcond=1e-4)
            except np.linalg.LinAlgError:
                dot_theta = np.zeros_like(b_hat)
                
            # 梯度裁剪
            max_step = 2.0
            grad_norm = np.linalg.norm(dot_theta)
            if grad_norm > max_step:
                dot_theta = dot_theta * (max_step / grad_norm)
            
            params = params + delta_tau * dot_theta
            params_history.append(params.copy())
            
            t1 = time.time()
            
            if step % 5 == 0 or step == n_steps - 1:
                print(f"SA-QITE Step {step}: Energy = {current_energy:.6f} (Time: {t1-t0:.4f}s)")
                
        return energies[-1], torch.tensor(params), energies, params_history

    def _compute_energy_exact(self, ansatz_func, params_np, H_matrix):
        with torch.no_grad():
            params_t = torch.tensor(params_np, dtype=torch.float64)
            qc = ansatz_func(params_t)
            state = self.backend.get_statevector(qc)
            # 使用 vdot 确保复数运算正确
            energy = torch.vdot(state, torch.matmul(H_matrix, state)).real.item()
        return energy

    def _compute_fidelity(self, ansatz_func, params1, params2):
        with torch.no_grad():
            p1 = torch.tensor(params1, dtype=torch.float64)
            p2 = torch.tensor(params2, dtype=torch.float64)
            state1 = self.backend.get_statevector(ansatz_func(p1))
            state2 = self.backend.get_statevector(ansatz_func(p2))
            # 确保归一化
            state1 = state1 / torch.norm(state1)
            state2 = state2 / torch.norm(state2)
            fid = torch.abs(torch.vdot(state1, state2))**2
        return fid.item()

    def _compute_spsa_qgt_sample(self, ansatz_func, params, epsilon):
        n_params = len(params)
        # SPSA 扰动方向
        delta1 = np.random.choice([-1, 1], size=n_params)
        delta2 = np.random.choice([-1, 1], size=n_params)
        
        # 四点法估计 FIM
        p_pp = params + epsilon * (delta1 + delta2)
        p_pm = params + epsilon * (delta1 - delta2)
        p_mp = params - epsilon * (delta1 + delta2)
        p_mm = params - epsilon * (delta1 - delta2)
        
        F1 = self._compute_fidelity(ansatz_func, params, p_pp)
        F2 = self._compute_fidelity(ansatz_func, params, p_pm)
        F3 = self._compute_fidelity(ansatz_func, params, p_mp)
        F4 = self._compute_fidelity(ansatz_func, params, p_mm)
        
        # 二阶差分近似
        delta_fidelity = (F1 - F2 - F3 + F4) / (8 * epsilon**2)
        
        term = np.outer(delta1, delta2) + np.outer(delta2, delta1)
        # FIM 约为 -1/2 * Hessian of Fidelity
        qgt_sample = -0.5 * term * delta_fidelity 
        return qgt_sample

    def _compute_spsa_grad_sample(self, ansatz_func, params, H_matrix, epsilon):
        n_params = len(params)
        delta = np.random.choice([-1, 1], size=n_params)
        
        p_plus = params + epsilon * delta
        p_minus = params - epsilon * delta
        
        E_plus = self._compute_energy_exact(ansatz_func, p_plus, H_matrix)
        E_minus = self._compute_energy_exact(ansatz_func, p_minus, H_matrix)
        
        grad_est = (E_plus - E_minus) / (2 * epsilon) * delta
        return grad_est