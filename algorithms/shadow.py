import numpy as np
import copy
from typing import List, Tuple
from backends.core import QuantumCircuit

class ClassicalShadow:
    """
    经典影子 (Classical Shadow) 实现
    用于通过随机测量高效构建量子态的经典描述。
    """

    def __init__(self, backend, circuit: QuantumCircuit, n_qubits: int):
        self.backend = backend
        self.circuit = circuit
        self.n_qubits = n_qubits
        self.snapshots = [] 

    def _append_rotation(self, qc: QuantumCircuit, axes: List[int]):
        """
        根据随机选择的基 (0:X, 1:Y, 2:Z) 添加旋转门，
        将测量基旋转到 Z 基。
        """
        for q, axis in enumerate(axes):
            if axis == 0:   # Measure in X: H gate -> Z
                qc.h(q)
            elif axis == 1: # Measure in Y: H S^dagger -> Z (近似实现: HS*)
                qc.sdg(q) 
                qc.h(q)
        return qc

    def collect(self, n_snapshots: int):
        """
        收集影子快照
        """
        self.snapshots = []

        for _ in range(n_snapshots):
            # 1. 随机生成每个比特的测量基 (0:X, 1:Y, 2:Z)
            axes = np.random.randint(0, 3, size=self.n_qubits)
            
            # 2. 复制原线路并添加旋转门
            run_qc = copy.deepcopy(self.circuit)
            self._append_rotation(run_qc, axes)
            
            # 3. 运行单次采样
            res = self.backend.run_sampling(run_qc, shots=1, measure_qubits=list(range(self.n_qubits)))
            
            # 4. 解析结果 
            bitstring = list(res.keys())[0]
            bits = [int(b) for b in bitstring]
            
            # 5. 存储 (测量结果 b, 测量基 P)
            self.snapshots.append((bits, axes))

    def estimate_observable(self, observable_pauli_string: str) -> float:
        """
        利用收集到的影子估计 Pauli 串的期望值 <P>.
        observable_pauli_string: 例如 "XYZ" 对应 Q0=X, Q1=Y, Q2=Z
        """
        if not self.snapshots:
            raise ValueError("请先调用 collect() 收集数据")

        # 将字符串 "XYZ" 转为数字 [0, 1, 2]
        map_c = {'X':0, 'Y':1, 'Z':2, 'I':-1}
        target_axes = [map_c[c] for c in observable_pauli_string]
        
        total_val = 0.0
        match_count = 0
        
        for (bits, axes) in self.snapshots:
            # 计算单个快照的迹 Tr(O \rho_hat)
            # 公式: \rho_hat = \bigotimes (3 U^dag |b><b| U - I)
            snapshot_val = 1.0
            for q in range(self.n_qubits):
                target_op = target_axes[q]
                
                if target_op == -1: 
                    continue
                
                measured_axis = axes[q]
                measured_bit = bits[q] 
                
                if measured_axis != target_op:
                    snapshot_val = 0.0
                    break
                else:
                    eigenval = 1 if measured_bit == 0 else -1
                    snapshot_val *= (3 * eigenval)
            
            total_val += snapshot_val
            
        return total_val / len(self.snapshots)