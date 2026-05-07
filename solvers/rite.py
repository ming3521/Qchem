import torch
import numpy as np
import time
from scipy.linalg import pinv
from .base import QuantumSolver

class RITESolver(QuantumSolver):
    """
    RITE (Randomized Imaginary Time Evolution) 
    """
    def solve(self, ansatz_func, init_params, hamiltonian, total_tau=1.0, delta_tau=0.1, 
                  n_samples=200, clifford_depth=2, regularization=1e-5):  
        params = init_params.clone().detach().to(dtype=torch.float64)
        params.requires_grad_(True)
        
        temp_qc = ansatz_func(params)
        n_qubits = temp_qc.n_qubits
        n_steps = int(total_tau / delta_tau)
        
        energies = []
        params_history = []
        
        print(f"--- RITE Start (Samples={n_samples}, Total Tau={total_tau}, dt={delta_tau}) ---")

        for step in range(n_steps):
            t0 = time.time()
            
            # 使用 RITE 策略计算 A 和 C
            A, C, curr_energy = self._compute_AC_rite(
                ansatz_func, params, hamiltonian, n_qubits, n_samples, clifford_depth
            )
            
            energies.append(curr_energy)
            
            A_np = A.numpy()
            C_np = C.numpy()
            
            # 尝试求解线性方程 A * dot_theta = C
            # 使用伪逆，并添加微小正则化以防奇异
            A_reg = A_np + regularization * np.eye(len(params))
            try:
                # 使用 lstsq 比直接 pinv 通常更数值稳定
                dot_theta, _, _, _ = np.linalg.lstsq(A_reg, C_np, rcond=1e-4)
            except np.linalg.LinAlgError:
                dot_theta = np.dot(pinv(A_reg), C_np)

            max_grad_norm = 10.0
            grad_norm = np.linalg.norm(dot_theta)
            if grad_norm > max_grad_norm:
                 dot_theta = dot_theta * (max_grad_norm / grad_norm)
            
            new_params = params.detach().numpy() + delta_tau * dot_theta
            params_history.append(new_params.copy())
            params = torch.tensor(new_params, dtype=torch.float64, requires_grad=True)
            
            t1 = time.time()
            # 增加打印频率
            if step % 2 == 0 or step == n_steps - 1:
                print(f"RITE Step {step}: Energy = {curr_energy:.6f} (Time: {t1-t0:.4f}s)")

        return energies[-1], params, energies, params_history

    def _generate_random_clifford_unitary(self, n_qubits, depth=2):
        """生成随机 Haar 酉矩阵 (模拟 2-design 性质)"""
        dim = 2 ** n_qubits
        # 使用 QR 分解生成均匀分布的随机酉矩阵
        rand_mat = torch.randn(dim, dim, dtype=torch.complex128)
        q, r = torch.linalg.qr(rand_mat)
        d = torch.diagonal(r)
        ph = d / torch.abs(d)
        q = q * ph
        return q

    def _compute_AC_rite(self, ansatz_func, params, hamiltonian, n_qubits, n_samples, clifford_depth):
        n_params = len(params)
        shift = np.pi / 2
        epsilon = 1e-6 
        
        with torch.no_grad():
            qc_0 = ansatz_func(params)
            state_0 = self.backend.get_statevector(qc_0)
            H_mat = self._pauli_string_to_matrix(n_qubits, hamiltonian)
            curr_energy = torch.vdot(state_0, torch.matmul(H_mat, state_0)).real.item()
            
            # 预计算移位后的态矢量
            psi_plus_list = []
            psi_minus_list = []
            for i in range(n_params):
                p_plus = params.clone(); p_plus[i] += shift
                p_minus = params.clone(); p_minus[i] -= shift
                psi_plus_list.append(self.backend.get_statevector(ansatz_func(p_plus)))
                psi_minus_list.append(self.backend.get_statevector(ansatz_func(p_minus)))

        qfim_sum = torch.zeros((n_params, n_params), dtype=torch.float64)
        
        # 批量采样计算 A 矩阵
        for k in range(n_samples):
            U = self._generate_random_clifford_unitary(n_qubits, depth=clifford_depth)
            
            u_state_0 = torch.matmul(U, state_0)
            probs_0 = torch.abs(u_state_0)**2
            probs_0 = torch.clamp(probs_0, min=epsilon) # 防止除零
            
            grads_ps = torch.zeros((n_params, 2**n_qubits), dtype=torch.float64)
            for i in range(n_params):
                u_psi_plus = torch.matmul(U, psi_plus_list[i])
                u_psi_minus = torch.matmul(U, psi_minus_list[i])
                
                prob_plus = torch.abs(u_psi_plus)**2
                prob_minus = torch.abs(u_psi_minus)**2
                
                grads_ps[i, :] = (prob_plus - prob_minus) / 2.0
            
            # RITE 估计器核心公式
            weighted_grads = grads_ps / probs_0 
            batch_qfim = torch.matmul(weighted_grads, grads_ps.T)
            qfim_sum += batch_qfim

        # 缩放因子
        scaling = 2 * (2**n_qubits + 1) / n_samples
        A = qfim_sum * scaling

        # 计算梯度向量 C
        C = torch.zeros(n_params, dtype=torch.float64)
        H_psi = torch.matmul(H_mat, state_0)
        
        for i in range(n_params):
            grad_psi = (psi_plus_list[i] - psi_minus_list[i]) / 2.0
            term1 = torch.vdot(grad_psi, H_psi)
            val = -torch.real(term1) 
            C[i] = val

        return A, C, curr_energy