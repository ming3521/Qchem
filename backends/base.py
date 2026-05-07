from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import torch
import numpy as np
try:
    from .core import QuantumCircuit
except ImportError:
    from core import QuantumCircuit

class QuantumBackend(ABC):
    """
    量子后端抽象基类
    """
    @abstractmethod
    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        """
        Type 1: 采样模式
        相当于真机接口。
        
        Args:
            circuit: 通用线路对象
            shots: 测量次数
            measure_qubits: 需要测量的比特索引列表
            
        Returns:
            Dict[str, int]: Bitstrings 字典, e.g., {'00': 50, '11': 50}
        """
        raise NotImplementedError("该后端不支持采样模式")

    @abstractmethod
    def get_statevector(self, circuit) -> Union[torch.Tensor, Any]:
        """
        Type 2: 状态矢量模式 (支持自动微分)
        
        Args:
            circuit: 通用线路对象 (包含参数)
            
        Returns:
            Tensor: 状态矢量，需保留计算图以支持反向传播
        """
        raise NotImplementedError("该后端不支持状态矢量/微分模式")






    
