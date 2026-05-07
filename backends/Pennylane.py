from .base import QuantumBackend
import pennylane as qml
import torch
from typing import Dict, List, Optional, Type
from .core import Compiler, QuantumCircuit

class PennyLaneBackend(QuantumBackend):
    """
    基于 PennyLane 的具体实现类
    """
    def __init__(self, device_name="default.qubit"):
        self.device_name = device_name
        # 建立门操作映射表：将 core.py 的指令翻译成 PennyLane 的函数
        self._gate_map = {
            'Id': qml.Identity,
            'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ,
            'H': qml.Hadamard, 'S': qml.S, 'T': qml.T,
            # 使用 lambda 表达式定义反向门
            'SDG': lambda wires: qml.adjoint(qml.S)(wires=wires),
            'TDG': lambda wires: qml.adjoint(qml.T)(wires=wires),
            'SX': qml.SX,
            'CNOT': qml.CNOT, 'CY': qml.CY, 'CZ': qml.CZ, 'SWAP': qml.SWAP,
            'CSWAP': qml.CSWAP, 'CCX': qml.Toffoli,
            'RX': qml.RX, 'RY': qml.RY, 'RZ': qml.RZ, 
            'Phase': qml.PhaseShift, 'U3': qml.U3,
            'CRX': qml.CRX, 'CRY': qml.CRY, 'CRZ': qml.CRZ, 
            'CPhase': qml.ControlledPhaseShift,
            'GlobalPhase': qml.GlobalPhase
        }
        self.compiler = BackendCompiler()
    def _apply_operation(self, inst):
        """内部辅助函数：执行单个指令"""
        # 1. 处理自定义矩阵
        if inst.matrix is not None:
            qml.QubitUnitary(inst.matrix, wires=inst.qubits)
            return

        # 2. 查找对应的 PennyLane 门
        gate_cls = self._gate_map.get(inst.name)
        if gate_cls is None:
            raise NotImplementedError(f"PennyLaneBackend 尚未支持门: {inst.name}")

        # 3. 执行门操作 (处理参数和控制位)
        if inst.control_qubits:
            # 如果有控制位，使用 qml.ctrl 包装
            qml.ctrl(
                 lambda: gate_cls(*inst.params, wires=inst.qubits), 
                 control=inst.control_qubits,
                 control_values=inst.control_values
            )()
        else:
            # 无控制位，直接执行
            gate_cls(*inst.params, wires=inst.qubits)

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        """实现采样接口"""
        
        dev = qml.device(self.device_name, wires=circuit.n_qubits, shots=shots)
        
        def circuit_func():
            for inst in circuit.instructions:
                self._apply_operation(inst)
            return qml.counts(wires=measure_qubits)

        qnode = qml.QNode(circuit_func, dev)
        result = qnode()
        
        # 4. 清洗数据：将 numpy 类型转换为标准的 python int 和 str
        # 比如 {np.str_('00'): np.int64(10)} -> {'00': 10}
        clean_result = {str(k): int(v) for k, v in result.items()}
        
        return clean_result

    def get_statevector(self, circuit) -> torch.Tensor:
        """实现状态矢量接口 (支持微分)"""
        # 动态创建设备 (shots=None 代表解析解)
        dev = qml.device(self.device_name, wires=circuit.n_qubits, shots=None)

        # 定义量子函数
        def circuit_func():
            for inst in circuit.instructions:
                self._apply_operation(inst)
            return qml.state()

        # 启用 PyTorch 接口
        qnode = qml.QNode(circuit_func, dev, interface='torch', diff_method='backprop')
        return qnode()
    

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

