import torch
import numpy as np
import time
from scipy.linalg import pinv
from .base import QuantumSolver

class VarQITESolver(QuantumSolver):
    """
    VarQITE (Variational Quantum Imaginary Time Evolution)
    """
    def solve(self, ansatz_func, init_params, hamiltonian, total_tau=1.0, delta_tau=0.05, regularization=1e-6, mode='autograd'):
        params = init_params.clone().detach().to(dtype=torch.float64)
        params.requires_grad_(True)
        
        temp_qc = ansatz_func(params)
        n_qubits = temp_qc.n_qubits
        n_steps = int(total_tau / delta_tau)
        energies = []
        params_history = []
        
        print(f"--- VarQITE Start (Mode={mode}, Total Tau={total_tau}, dt={delta_tau}) ---")

        for step in range(n_steps):
            t0 = time.time()
            if mode == 'autograd':
                A, C, curr_energy = self._compute_AC_autograd(ansatz_func, params, hamiltonian, n_qubits)
            elif mode == 'parameter_shift':
                A, C, curr_energy = self._compute_AC_parameter_shift(ansatz_func, params, hamiltonian, n_qubits)
            elif mode == 'hadamard': # 新增模式
                A, C, curr_energy = self._compute_AC_hadamard(ansatz_func, params, hamiltonian, n_qubits)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            energies.append(curr_energy)
            A_np = A.numpy() + regularization * np.eye(len(params))
            C_np = C.numpy()
            dot_theta = np.dot(pinv(A_np), C_np)
            
            new_params = params.detach().numpy() + delta_tau * dot_theta
            params_history.append(new_params.copy())
            params = torch.tensor(new_params, dtype=torch.float64, requires_grad=True)
            
            t1 = time.time()
            if step % 10 == 0 or step == n_steps - 1:
                print(f"Step {step}: Energy = {curr_energy:.6f} (Time: {t1-t0:.4f}s)")

        return energies[-1], params, energies, params_history

    def _compute_AC_autograd(self, ansatz_func, params, hamiltonian, n_qubits):
        def get_psi_real(p):
            return torch.real(self.backend.get_statevector(ansatz_func(p)))
        def get_psi_imag(p):
            return torch.imag(self.backend.get_statevector(ansatz_func(p)))
        J_real = torch.autograd.functional.jacobian(get_psi_real, params)
        J_imag = torch.autograd.functional.jacobian(get_psi_imag, params)
        J = J_real + 1j * J_imag
        A = torch.real(torch.matmul(J.conj().T, J))
        
        with torch.no_grad():
            qc = ansatz_func(params)
            state = self.backend.get_statevector(qc)
            H_matrix = self._pauli_string_to_matrix(n_qubits, hamiltonian)
            curr_energy = torch.vdot(state, torch.matmul(H_matrix, state)).real.item()
            H_state = torch.matmul(H_matrix, state)
        
        C = -torch.real(torch.matmul(J.conj().T, H_state))
        return A, C, curr_energy

    def _compute_AC_parameter_shift(self, ansatz_func, params, hamiltonian, n_qubits):
        n_params = len(params)
        shift = np.pi / 2
        with torch.no_grad():
            qc = ansatz_func(params)
            state_0 = self.backend.get_statevector(qc)
            H_matrix = self._pauli_string_to_matrix(n_qubits, hamiltonian)
            curr_energy = torch.vdot(state_0, torch.matmul(H_matrix, state_0)).real.item()
            H_state = torch.matmul(H_matrix, state_0)

        grads = []
        for i in range(n_params):
            p_plus = params.clone(); p_plus[i] += shift
            p_minus = params.clone(); p_minus[i] -= shift
            with torch.no_grad():
                psi_plus = self.backend.get_statevector(ansatz_func(p_plus))
                psi_minus = self.backend.get_statevector(ansatz_func(p_minus))
            grad_i = (psi_plus - psi_minus) / 2.0 
            grads.append(grad_i)
            
        A = torch.zeros((n_params, n_params), dtype=torch.float64)
        C = torch.zeros(n_params, dtype=torch.float64)
        for i in range(n_params):
            C[i] = -torch.real(torch.vdot(grads[i], H_state))
            for j in range(i, n_params):
                val = torch.real(torch.vdot(grads[i], grads[j]))
                A[i, j] = val
                A[j, i] = val 
        return A, C, curr_energy

    def _compute_AC_hadamard(self, ansatz_func, params, hamiltonian, n_qubits):
        """
        使用哈达玛测试逻辑计算 A 矩阵和 C 向量。
        """
        n_params = len(params)
        aux_wire = n_qubits  # 辅助比特索引
        total_qubits = n_qubits + 1
        
        # 1. 计算当前能量 (用于返回)
        with torch.no_grad():
            qc_base = ansatz_func(params)
            state_0 = self.backend.get_statevector(qc_base)
            H_matrix = self._pauli_string_to_matrix(n_qubits, hamiltonian)
            curr_energy = torch.vdot(state_0, torch.matmul(H_matrix, state_0)).real.item()

        A = torch.zeros((n_params, n_params), dtype=torch.float64)
        C = torch.zeros(n_params, dtype=torch.float64)

        # 模拟哈达玛测试电路逻辑
        # 在真实硬件中，这里会对应不同的 QuantumScript 批处理
        for i in range(n_params):
            # --- 计算 C[i] (对应 Im <psi| Gi^dagger H |psi>) ---
            # 逻辑：H(aux) -> Ctrl-Gi(target) -> H(aux) -> 测量 aux 上的 Y 算符
            # 这里我们利用后端模拟该过程：
            Gi = self._get_generator_matrix_for_param(ansatz_func, params, i, n_qubits)
            
            # 构造 C 的测量值：-Re <psi | (i/2 Gi) H | psi> = 1/2 Im <psi | Gi H | psi>
            # 对应哈达玛测试测量辅助比特 Y 的结果
            val_c = 0.5 * torch.vdot(state_0, torch.matmul(Gi, torch.matmul(H_matrix, state_0))).imag
            C[i] = val_c

            # --- 计算 A[i, j] (对应 Re <partial_i psi | partial_j psi>) ---
            for j in range(i, n_params):
                Gj = self._get_generator_matrix_for_param(ansatz_func, params, j, n_qubits)
                
                # A_ij = 1/4 Re <psi | Gi^dagger Gj | psi>
                # 对应哈达玛测试中两个生成元相继作用后测量辅助比特 Z 的结果
                val_a = 0.25 * torch.vdot(state_0, torch.matmul(Gi.conj().T, torch.matmul(Gj, state_0))).real
                A[i, j] = val_a
                A[j, i] = val_a

        return A, C, curr_energy

    def _get_generator_matrix_for_param(self, ansatz_func, params, param_idx, n_qubits):
        """
        辅助函数：获取特定参数对应的生成元矩阵 G。
        在 PennyLane 中，这对应 op.generator()。
        """
        # 这是一个简化实现，假设每个参数对应一个 Pauli 生成元
        # 实际移植时，需要根据你的 ansatz 定义提取对应的 Pauli 算符
        # 这里演示如何通过 autograd 提取算子矩阵
        p = params.clone().detach()
        p.requires_grad_(True)
        qc = ansatz_func(p)
        # 提取第 param_idx 个门对应的生成元 (通常为 Pauli 矩阵)
        # 这里为了演示，使用数值微扰近似提取生成元矩阵
        eps = 1e-4
        p_plus = p.clone(); p_plus[param_idx] += eps
        state_p = self.backend.get_statevector(ansatz_func(p))
        state_plus = self.backend.get_statevector(ansatz_func(p_plus))
        # 根据 |d_psi> = -i/2 G |psi> 推导 G
        diff = (state_plus - state_p) / eps
        # G * state = 2j * diff
        # 简化返回（实际应用中应从 ansatz 结构中直接读取 Pauli 类型）
        return self._pauli_string_to_matrix(n_qubits, [("Z", [0])]) # 示例占位


