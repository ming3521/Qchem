from abc import ABC, abstractmethod
import torch
from backends.core import QuantumCircuit

class Ansatz(ABC):
    """
    Ansatz 基类
    规定了所有拟设类必须实现 forward 方法，输入参数，输出量子线路。
    """
    def __init__(self, n_qubits: int, n_params: int = 0):
        self.n_qubits = n_qubits
        self.n_params = n_params

    @abstractmethod
    def forward(self, params: torch.Tensor) -> QuantumCircuit:
        """
        构建参数化量子线路
        Args:
            params (torch.Tensor): 形状为 (n_params,) 的参数张量
        Returns:
            QuantumCircuit: 构建好的线路
        """
        pass

    def __call__(self, params: torch.Tensor) -> QuantumCircuit:
        """允许像函数一样调用: circuit = ansatz(params)"""
        return self.forward(params)
    
    @property
    def num_parameters(self):
        return self.n_params