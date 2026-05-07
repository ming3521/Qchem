import sys
import torch
import numpy as np
from openfermion import FermionOperator, jordan_wigner
from .base import Ansatz
from backends.core import QuantumCircuit

class UCCSD(Ansatz):
    """
    Unitary Coupled Cluster Singles and Doubles (UCSSD) Ansatz.
    使用 Jordan-Wigner 变换将费米子激发算符映射为 qubit 算符，
    并使用一阶 Trotter 分解构建量子线路。
    """
    def __init__(self, n_qubits, n_electrons, trotter_steps=1):
        """
        Args:
            n_qubits: 总量子比特数 (即总自旋轨道数)
            n_electrons: 电子数量
            trotter_steps: Trotter 分解的阶数 (通常设为 1 即可)
        """
        self.n_electrons = n_electrons
        self.trotter_steps = trotter_steps
        
        # 1. 确定占据轨道 (Occ) 和 虚轨道 (Vir) 的索引
        # 在 Hartree-Fock 态中，前 n_electrons 个轨道是被占据的
        self.occupied_indices = list(range(n_electrons))
        self.virtual_indices = list(range(n_electrons, n_qubits))
        
        # 2. 生成所有合法的激发项 (Fermionic Operators)
        # 存储的是 (generator_pauli_strings, parameter_index)
        # generator = T - T_dagger
        self.excitation_ops = self._generate_excitations()
        
        # 参数数量等于激发项的数量
        super().__init__(n_qubits, n_params=len(self.excitation_ops))
        
    def _generate_excitations(self):
        """生成单激发和双激发的 Pauli 串列表"""
        excitations_pauli_list = []
        
        #  单激发 (Singles): i (occ) -> a (vir) 
        for i in self.occupied_indices:
            for a in self.virtual_indices:
                # 算符: a^\dagger_a a_i - a^\dagger_i a_a
                # openfermion 格式: "a^ i" 表示 a_i^\dagger
                # 这是一个反厄米算符
                op = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                
                # 映射到 Pauli 串 (使用 Jordan-Wigner)
                qubit_op = jordan_wigner(op)
                
                # 提取其中的每一项 (coef * PauliString)
                # UCCSD 的一项通常由多个 Pauli 串组成，它们共用一个参数 theta
                # e.g. exp(theta * (XY - YX))
                terms = []
                for term, coeff in qubit_op.terms.items():
                    # term 是一个 tuple list: [(0, 'X'), (1, 'Y')]
                    # coeff 是系数 (对于 T-Tdag 通常是纯虚数，但在 exponent 中变为实数旋转)
                    real_coeff = coeff.imag 
                    if abs(real_coeff) > 1e-8:
                        terms.append((term, real_coeff))
                
                if terms:
                    excitations_pauli_list.append(terms)

        #  双激发 (Doubles): i, j (occ) -> a, b (vir) 
        for i_idx, i in enumerate(self.occupied_indices):
            for j in self.occupied_indices[i_idx+1:]:
                for a_idx, a in enumerate(self.virtual_indices):
                    for b in self.virtual_indices[a_idx+1:]:
                        # 算符: a^\dagger_a a^\dagger_b a_j a_i - h.c.
                        op = FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0))) - \
                             FermionOperator(((i, 1), (j, 1), (b, 0), (a, 0)))
                        
                        qubit_op = jordan_wigner(op)
                        terms = []
                        for term, coeff in qubit_op.terms.items():
                            real_coeff = coeff.imag
                            if abs(real_coeff) > 1e-8:
                                terms.append((term, real_coeff))
                        
                        if terms:
                            excitations_pauli_list.append(terms)
                            
        return excitations_pauli_list

    def forward(self, params: torch.Tensor) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, name="UCCSD")
        
        # 1. 制备 Hartree-Fock 初始态 (占据态置 1)
        for idx in self.occupied_indices:
            qc.x(idx)
            
        # 2. 应用激发演化 (Trotter Steps)
        # U = prod_k exp(theta_k * (T_k - T_k^\dagger))
        
        for _ in range(self.trotter_steps):
            # 遍历每一个激发项 (由一个参数控制)
            for param_idx, pauli_terms in enumerate(self.excitation_ops):
                theta = params[param_idx]
                
                # 遍历构成该激发项的每一个 Pauli 串
                # e.g. double excitation 可能会产生 8 个 Pauli 串
                for pauli_string, coeff in pauli_terms:
                    angle = theta * coeff 
                    self._append_pauli_evolution(qc, pauli_string, angle)
                    
        return qc

    def _append_pauli_evolution(self, qc, pauli_string, angle):
        """
        在电路中添加 exp(i * angle * P) 演化门
        P 是 Pauli String, e.g. [(0, 'X'), (1, 'Z')] 表示 X0 Z1
        实现逻辑：CNOT  + RZ
        """
        # 如果是空串 (Identity)，跳过
        if not pauli_string:
            return

        qubits = [idx for idx, _ in pauli_string]
        ops = [op for _, op in pauli_string]
        
        # 1. 基变换 (Basis Change) -> 变到 Z 基
        for q, op in zip(qubits, ops):
            if op == 'X':
                qc.h(q)
            elif op == 'Y':
                qc.rx(q, np.pi/2) 
                
        # 2. CNOT  (Compute Parity)
        # 将 parity 算到最后一个比特上
        for i in range(len(qubits) - 1):
            qc.cx(qubits[i], qubits[i+1])
            
        # 3. RZ 旋转 (Evolution)
        last_qubit = qubits[-1]
        qc.rz(last_qubit, -2.0 * angle)
        
        # 4. 反向 CNOT  (Uncompute Parity)
        for i in range(len(qubits) - 2, -1, -1):
            qc.cx(qubits[i], qubits[i+1])
            
        # 5. 反向基变换
        for q, op in zip(qubits, ops):
            if op == 'X':
                qc.h(q)
            elif op == 'Y':
                qc.rx(q, -np.pi/2)