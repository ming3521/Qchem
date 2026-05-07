import torch
import numpy as np
from ansatz.base import Ansatz, QuantumCircuit 

class HEAAnsatz(Ansatz):
    """
    硬件高效拟设 (HEA) 
    直接调用 QuantumCircuit 类来构建线路
    """
    def __init__(self, n_qubits, depth=1):
        # 计算总参数量: 每层 n_qubits 个旋转 + 最后一层 n_qubits 个旋转
        # Total params = n_qubits * (depth + 1)
        n_params = n_qubits * (depth + 1)
        super().__init__(n_qubits, n_params)
        self.depth = depth
    
    def forward(self, params):
        if params.shape[0] != self.n_params:
             raise ValueError(f"参数数量错误: 需要 {self.n_params}, 实际 {params.shape[0]}")

        # 重塑以便操作: [depth + 1, n_qubits]
        reshaped_params = params.view(self.depth + 1, self.n_qubits)
        
        qc = QuantumCircuit(self.n_qubits)
        
        # 2. 搭建层级
        for d in range(self.depth):
            # A. 旋转层 Ry
            for i in range(self.n_qubits):
                qc.ry(i, reshaped_params[d, i])
            
            # B. 纠缠层 CNOT (线性链)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            
        # 3. 最后一层旋转
        for i in range(self.n_qubits):
            qc.ry(i, reshaped_params[self.depth, i])
            
        return qc