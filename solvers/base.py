import sys
import torch
import numpy as np
from abc import ABC, abstractmethod
from backends.base import QuantumBackend

class QuantumSolver(ABC):
    """
    量子算法求解器基类 (Abstract Base Class)
    提供基础的量子门定义、Pauli串转换工具等通用功能。
    """
    def __init__(self, backend: QuantumBackend):
        self.backend = backend
        # 预定义基础门矩阵 (complex128)
        self._init_gate_matrices()

    def _init_gate_matrices(self):
        """初始化基础量子门矩阵"""
        self.I = torch.eye(2, dtype=torch.complex128)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        self.H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
        self.S = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex128)
        self.Sdg = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex128)
        
        # 映射表
        self.pauli_map = {'I': self.I, 'X': self.X, 'Y': self.Y, 'Z': self.Z}

    def _pauli_string_to_matrix(self, n_qubits, pauli_data):
        """通用工具：将 Pauli 算符转换为矩阵 (Tensor)"""
        def get_term_matrix(p_str):
            if len(p_str) != n_qubits:
                 pass # 实际应用中应做对齐检查
            mat = torch.tensor([1.0], dtype=torch.complex128)
            for char in p_str:
                mat = torch.kron(mat, self.pauli_map[char])
            return mat

        if isinstance(pauli_data, list):
            total_mat = torch.zeros((2**n_qubits, 2**n_qubits), dtype=torch.complex128)
            for coeff, p_str in pauli_data:
                total_mat += coeff * get_term_matrix(p_str)
            return total_mat
        elif isinstance(pauli_data, str):
            return get_term_matrix(pauli_data)
        else:
            raise TypeError("Input must be a list of (coeff, string) or a single string")

    @abstractmethod
    def solve(self, *args, **kwargs):
        """所有子类必须实现求解方法"""
        pass