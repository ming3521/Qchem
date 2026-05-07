from .base import QuantumBackend
import torch
from typing import Dict, List, Optional, Type
import numpy as np
from .core import Compiler, QuantumCircuit

class QiskitBackend(QuantumBackend):
    """
    基于 Qiskit 的后端实现
    """
    def __init__(self, device_name="aer_simulator"):
        self.device_name = device_name
        self._simulator = None
        self.compiler = BackendCompiler()
        # 尝试加载 Qiskit 模拟器
        # 优先级: Qiskit Aer (高性能) -> BasicSimulator (Qiskit 1.0+ 内置) -> BasicAer (旧版内置)
        try:
            from qiskit_aer import AerSimulator
            self._simulator = AerSimulator()
        except ImportError:
            try:
                # 适配 Qiskit 1.0+
                from qiskit.providers.basic_provider import BasicSimulator
                self._simulator = BasicSimulator()
            except ImportError:
                try:
                    # 适配 Qiskit < 1.0
                    from qiskit import BasicAer
                    self._simulator = BasicAer.get_backend('qasm_simulator')
                except ImportError:
                    pass #将在运行时报错

    def _check_qiskit(self):
        if self._simulator is None:
            raise ImportError("Qiskit backend requires 'qiskit' (and optionally 'qiskit-aer'). Please install them.")

    def run_sampling(self, circuit, shots: int, measure_qubits: List[int]) -> Dict[str, int]:
        """
        采样模式: 转换电路 -> 添加测量 -> 运行模拟器 -> 格式化结果
        """
        self._check_qiskit()
        from qiskit import ClassicalRegister, transpile
        
        # 1. 转换为 Qiskit 电路
        qc_qiskit = circuit.to_qiskit()
        
        # 2. 添加经典寄存器和测量指令
        # measure_qubits 列表中的第 i 个比特将被测量到第 i 个经典比特上
        num_meas = len(measure_qubits)
        creg = ClassicalRegister(num_meas, name='c')
        qc_qiskit.add_register(creg)
        
        # Qiskit 支持批量测量: measure([q_idx...], [c_idx...])
        # 注意：这里我们按照 measure_qubits 的顺序一一映射到 c[0], c[1]...
        qc_qiskit.measure(measure_qubits, range(num_meas))
        
        # 3. 编译与运行
        # 某些 backend (如 Aer) 需要 transpile
        transpiled_qc = transpile(qc_qiskit, self._simulator)
        result = self._simulator.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
        
        # 4. 格式化结果
        # Qiskit 的 bitstring 是 Little-Endian (qn...q0)，且与寄存器顺序相关
        # 我们测量映射为: measure_qubits[0] -> c0, measure_qubits[1] -> c1
        # Qiskit 返回的 key 是 "c_last ... c1 c0"
        # 我们需要的格式通常是 "val(measure_qubits[0]) val(measure_qubits[1])..." 即 "c0 c1 ..."
        # 因此需要将 key 字符串反转
        
        formatted_counts = {}
        for bitstring, count in counts.items():
            # 反转字符串: "10" (c1=1, c0=0) -> "01" (c0=0, c1=1) -> 对应 (q_meas_0=0, q_meas_1=1)
            new_key = bitstring[::-1]
            formatted_counts[new_key] = count
            
        return formatted_counts

    def get_statevector(self, circuit) -> torch.Tensor:
        """
        解析模式: 计算理论状态矢量
        注意: Qiskit 原生不支持 PyTorch 自动微分，此模式主要用于获取数值解。
        """
        try:
            from qiskit.quantum_info import Statevector
        except ImportError:
            raise ImportError("Method requires qiskit.")

        # 1. 转换为 Qiskit 电路
        qc_qiskit = circuit.to_qiskit()
        
        # 2. 计算状态矢量
        # Statevector.from_instruction 会自动模拟电路演化
        sv = Statevector.from_instruction(qc_qiskit)
        
        # 3. 转换为 Torch Tensor
        # sv.data 是 numpy array (complex128)
        tensor_state = torch.tensor(np.array(sv.data), dtype=torch.complex128)
        
        return tensor_state
    

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