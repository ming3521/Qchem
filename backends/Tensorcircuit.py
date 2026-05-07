from .base import QuantumBackend
import torch
from typing import Dict, List, Optional, Type
from .core import Compiler, QuantumCircuit
try:
    import tensorcircuit as tc
    # 设置默认后端为 pytorch，精度为 complex128
    tc.set_backend("pytorch")
    tc.set_dtype("complex128")
    TC_AVAILABLE = True
except ImportError:
    TC_AVAILABLE = False
    
class TensorCircuitBackend(QuantumBackend):
    """
    基于 TensorCircuit 的后端实现
    支持 PyTorch 自动微分，适合 VarQITE 等需要计算梯度的算法
    """
    def __init__(self):
        if not TC_AVAILABLE:
            raise ImportError("请先安装 tensorcircuit: pip install tensorcircuit")
        
        # 门名称映射表
        self._gate_map = {
            'Id': 'i', 'X': 'x', 'Y': 'y', 'Z': 'z',
            'H': 'h', 'S': 's', 'T': 't',
            'CNOT': 'cnot', 'CX': 'cnot', 'SWAP': 'swap',
            'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
            'Phase': 'phase', 'U3': 'u3',
            'CZ': 'cz'
        }
        self.compiler = BackendCompiler()

    def _apply_operation(self, c: tc.Circuit, inst):
        """将 instruction 应用到 tc.Circuit"""
        # 1. 处理自定义矩阵
        if inst.matrix is not None:
            c.unitary(inst.qubits[0], unitary=inst.matrix, name="Custom")
            return

        # 2. 获取门名称
        tc_gate_name = self._gate_map.get(inst.name)
        if tc_gate_name is None:
            # 尝试直接使用小写名称
            tc_gate_name = inst.name.lower()
            if not hasattr(c, tc_gate_name):
                 raise NotImplementedError(f"TensorCircuitBackend 尚未支持门: {inst.name}")

        # 3. 执行门操作
        # TensorCircuit 调用格式: c.gate(idx, theta=val) 或 c.gate(idx1, idx2)
        if inst.params:
            if inst.name in ['RX', 'RY', 'RZ', 'Phase']:
                 getattr(c, tc_gate_name)(*inst.qubits, theta=inst.params[0])
            elif inst.name == 'U3':
                 getattr(c, tc_gate_name)(*inst.qubits, theta=inst.params[0], phi=inst.params[1], lam=inst.params[2])
            else:
                 # 其他带参门
                 getattr(c, tc_gate_name)(*inst.qubits, *inst.params)
        else:
            # 无参数门
            getattr(c, tc_gate_name)(*inst.qubits)

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        c = tc.Circuit(circuit.n_qubits)
        for inst in circuit.instructions:
            self._apply_operation(c, inst)
        
        # 采样 (batch=shots)
        samples = c.sample(allow_state=True, batch=shots, format="sample_bin")
        
        # 统计计数
        counts = {}
        for s in samples:
            # 提取需要的 measure_qubits 位
            s_str = "".join([str(int(s[i])) for i in measure_qubits])
            counts[s_str] = counts.get(s_str, 0) + 1
        return counts

    def get_statevector(self, circuit) -> torch.Tensor:
        """
        返回状态矢量 (PyTorch Tensor)，支持自动微分
        """
        c = tc.Circuit(circuit.n_qubits)
        for inst in circuit.instructions:
            self._apply_operation(c, inst)
        # 返回 state (Tensor)
        return c.state()
    

class BackendCompiler:
    """
    TensorCircuitBackend 暴露出来的编译工具包。

    这里只做一层很薄的包装，真正的编译逻辑仍然复用同级 core.py 中的 Compiler，
    避免 core 和 backend 中出现两套不一致的编译实现。
    """

    def to_qiskit(self, circuit: QuantumCircuit):
        return Compiler.to_qiskit(circuit)

    def from_qiskit(self, qiskit_qc, circuit_cls: Type[QuantumCircuit] = QuantumCircuit):
        return Compiler.from_qiskit(qiskit_qc, circuit_cls)

    def transpile(
        self,
        circuit: QuantumCircuit,
        basis_gates: Optional[List[str]] = None,
        optimization_level: int = 1,
    ) -> QuantumCircuit:
        return Compiler.transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )

    def to_qasm(self, circuit: QuantumCircuit) -> str:
        return Compiler.to_qasm(circuit)

    def from_qasm(self, qasm_str: str, circuit_cls: Type[QuantumCircuit] = QuantumCircuit):
        return Compiler.from_qasm(qasm_str, circuit_cls)

    def info(self, circuit: QuantumCircuit) -> Dict:
        return Compiler.get_circuit_info(circuit)

    def draw(self, circuit: QuantumCircuit, output: str = "text", **kwargs):
        return Compiler.draw(circuit, output=output, **kwargs)

    