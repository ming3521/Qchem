from typing import List, Optional, Union
import numpy as np
import os
import qiskit
class Instruction:
    def __init__(self, name=None, matrix=None, circuit=None, 
                 qubits=None, params=None, n_qubits_limit=None):
        self.name = name
        self.matrix = matrix
        self.circuit = circuit
        self.qubits = list(qubits) if qubits else []
        
        # 参数列表
        self.params = params if params is not None else []
        
        # 线路总比特数限制
        self._n_qubits_limit = n_qubits_limit
        
        # 存储控制信息
        self.control_qubits: List[int] = [] 
        self.control_values: List[int] = [] 

    def control(self, control_qubits: Union[int, List[int]], control_values: Optional[List[int]] = None):
        """添加控制位"""
        
        #  1. 格式归一化 
        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]
        # 支持 range/tuple
        if not isinstance(control_qubits, list):
            control_qubits = list(control_qubits)
            
        if control_values is None:
            control_values = [1] * len(control_qubits)
        else:
            if not isinstance(control_values, list):
                 raise TypeError("control_values 必须是 list")
            if len(control_qubits) != len(control_values):
                raise ValueError(f"Control qubits ({len(control_qubits)}) and values ({len(control_values)}) mismatch")
            if any(v not in (0, 1) for v in control_values):
                raise ValueError("control_values 只能包含 0 或 1")

        #  2. 合法性校验 
        # A. 检查越界
        if self._n_qubits_limit is not None:
            for q in control_qubits:
                if q < 0 or q >= self._n_qubits_limit:
                    raise IndexError(f"Control qubit {q} is out of range [0, {self._n_qubits_limit-1}]")

        # B. 检查重叠
        target_set = set(self.qubits)
        control_set = set(control_qubits)
        
        intersection = target_set.intersection(control_set)
        if intersection:
            raise ValueError(f"Control qubits {intersection} overlap with target qubits {self.qubits}")
            
        existing_control_set = set(self.control_qubits)
        intersection_existing = existing_control_set.intersection(control_set)
        if intersection_existing:
             raise ValueError(f"Control qubits {intersection_existing} are already assigned")

        #  3. 存储 
        self.control_qubits.extend(control_qubits)
        self.control_values.extend(control_values)
        return self

    def __repr__(self):
        # 1. 类型描述
        if self.circuit:
            c_name = getattr(self.circuit, 'name', 'SubCircuit')
            type_desc = f"Circuit(name='{c_name}')"
        elif self.matrix is not None:
            shape = self.matrix.shape if hasattr(self.matrix, 'shape') else 'Custom'
            type_desc = f"Matrix(shape={shape})"
        else:
            type_desc = f"Gate(name='{self.name}')"

        # 2. 参数描述
        params_str = ""
        if self.params:
            p_list = [f"{p:.3f}" if isinstance(p, float) else str(p) for p in self.params]
            params_str = f", params={p_list}"

        # 3. 控制描述
        ctrl_str = ""
        if self.control_qubits:
            ctrl_map = {q: v for q, v in zip(self.control_qubits, self.control_values)}
            ctrl_str = f", controls={ctrl_map}"

        return f"<Instruction: {type_desc}, qubits={self.qubits}{params_str}{ctrl_str}>"


class QuantumCircuit:
    def __init__(self, n_qubits: int, name: str = None):
        self.n_qubits = n_qubits
        self.instructions = [] 
        self.name = name

    def _check_qubits(self, qubits: List[int]):
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise IndexError(f"Qubit index {q} out of range [0, {self.n_qubits-1}]")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Duplicate qubits in instruction: {qubits}")

    def _is_unitary(self, matrix: np.ndarray, tol=1e-9) -> bool:
        """校验矩阵是否为酉矩阵"""
        m = np.array(matrix)
        if m.shape[0] != m.shape[1]:
            return False 
        u_u_dagger = m @ m.conj().T
        return np.allclose(u_u_dagger, np.eye(m.shape[0]), atol=tol)

    # 1. Append
    def append(self, gate_name: str, qubits: Union[int, List[int]], params=None):
        if isinstance(qubits, int): qubits = [qubits]
        if not isinstance(qubits, list): qubits = list(qubits)
            
        self._check_qubits(qubits)
        
        inst = Instruction(name=gate_name, qubits=qubits, params=params, n_qubits_limit=self.n_qubits)
        self.instructions.append(inst)
        return inst

    # 2. Unitary
    def unitary(self, matrix: np.ndarray, qubits: Union[int, List[int]], check_unitary=True):
        if isinstance(qubits, int): qubits = [qubits]
        if not isinstance(qubits, list): qubits = list(qubits)
        self._check_qubits(qubits)
        
        matrix = np.array(matrix)
        expected_dim = 2 ** len(qubits)
        
        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"Matrix shape {matrix.shape} mismatch for {len(qubits)} qubits")
        
        if check_unitary:
            if not self._is_unitary(matrix):
                 raise ValueError("Provided matrix is NOT unitary.")

        inst = Instruction(matrix=matrix, qubits=qubits, n_qubits_limit=self.n_qubits)
        self.instructions.append(inst)
        return inst

    # 3. Extend
    def extend(self, sub_qc: 'QuantumCircuit', qubits_mapping: List[int]):
        if not isinstance(qubits_mapping, list): qubits_mapping = list(qubits_mapping)
        self._check_qubits(qubits_mapping)
        if len(qubits_mapping) != sub_qc.n_qubits:
            raise ValueError("Mapping length mismatch")

        inst = Instruction(circuit=sub_qc, qubits=qubits_mapping, n_qubits_limit=self.n_qubits)
        self.instructions.append(inst)
        return inst

    # ==========================================
    #           0.  全局门
    # ==========================================

    def global_phase(self, theta):
        return self.append('GlobalPhase', [], params=[theta])
    
    # ==========================================
    #           1. 单比特门 (无参数) - 支持列表/Range输入
    # ==========================================

    def id(self, qubit):
        if isinstance(qubit, (list, tuple, range)): return [self.id(q) for q in qubit]
        return self.append('Id', [qubit])
    
    def x(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.x(q) for q in qubit]
        return self.append('X', [qubit])
        
    def y(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.y(q) for q in qubit]
        return self.append('Y', [qubit])
        
    def z(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.z(q) for q in qubit]
        return self.append('Z', [qubit])
    
    def h(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.h(q) for q in qubit]
        return self.append('H', [qubit])
        
    def s(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.s(q) for q in qubit]
        return self.append('S', [qubit])
        
    def sdg(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.sdg(q) for q in qubit]
        return self.append('SDG', [qubit])
        
    def t(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.t(q) for q in qubit]
        return self.append('T', [qubit])
        
    def tdg(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.tdg(q) for q in qubit]
        return self.append('TDG', [qubit])
        
    def sx(self, qubit): 
        if isinstance(qubit, (list, tuple, range)): return [self.sx(q) for q in qubit]
        return self.append('SX', [qubit])

    # ==========================================
    #           2. 单比特门 (带参数) - 支持列表输入
    # ==========================================
    
    def rx(self, qubit, theta): 
        if isinstance(qubit, (list, tuple, range)): return [self.rx(q, theta) for q in qubit]
        return self.append('RX', [qubit], params=[theta])
        
    def ry(self, qubit, theta): 
        if isinstance(qubit, (list, tuple, range)): return [self.ry(q, theta) for q in qubit]
        return self.append('RY', [qubit], params=[theta])
        
    def rz(self, qubit, theta): 
        if isinstance(qubit, (list, tuple, range)): return [self.rz(q, theta) for q in qubit]
        return self.append('RZ', [qubit], params=[theta])
    
    def p(self, qubit, theta): 
        if isinstance(qubit, (list, tuple, range)): return [self.p(q, theta) for q in qubit]
        return self.append('Phase', [qubit], params=[theta])
        
    def u3(self, qubit, theta, phi, lam):
        if isinstance(qubit, (list, tuple, range)): return [self.u3(q, theta, phi, lam) for q in qubit]
        return self.append('U3', [qubit], params=[theta, phi, lam])

    # ==========================================
    #           3. 双比特门
    # ==========================================
    
    def cx(self, control, target): return self.append('CNOT', [control, target])
    def cy(self, control, target): return self.append('CY', [control, target])
    def cz(self, control, target): return self.append('CZ', [control, target])
    def ch(self, control, target): return self.append('CH', [control, target])
    
    def swap(self, q1, q2): return self.append('SWAP', [q1, q2])
    def iswap(self, q1, q2): return self.append('iSWAP', [q1, q2])
    def dcx(self, q1, q2): return self.append('DCX', [q1, q2]) 
    def ecr(self, q1, q2): return self.append('ECR', [q1, q2]) 

    # ==========================================
    #           4. 双比特门 (带参数)
    # ==========================================
    
    def cp(self, control, target, theta): return self.append('CPhase', [control, target], params=[theta])
    def crx(self, control, target, theta): return self.append('CRX', [control, target], params=[theta])
    def cry(self, control, target, theta): return self.append('CRY', [control, target], params=[theta])
    def crz(self, control, target, theta): return self.append('CRZ', [control, target], params=[theta])
    
    def rxx(self, q1, q2, theta): return self.append('RXX', [q1, q2], params=[theta])
    def ryy(self, q1, q2, theta): return self.append('RYY', [q1, q2], params=[theta])
    def rzz(self, q1, q2, theta): return self.append('RZZ', [q1, q2], params=[theta])
    def rxy(self, q1, q2, theta): return self.append('RXY', [q1, q2], params=[theta])
        
    # ==========================================
    #           5. 三比特及其他门
    # ==========================================
    
    def ccx(self, c1, c2, target): return self.append('CCX', [c1, c2, target])
    def cswap(self, control, q1, q2): return self.append('CSWAP', [control, q1, q2])

    # ==========================================
    #           6. 多比特门
    # ==========================================
    def mrz(self, qubits, theta): return self.append('MRZ', list(qubits), params=[theta])
    def mcx(self, controls, target): return self.append('MCX', list(controls)+[target])
    def mcrz(self, controls, target, theta): return self.append('MCRZ', list(controls)+[target], params=[theta])

    # ==========================================
    #           7. 编译与转换接口 
    # ==========================================

    def to_qiskit(self):
        """转换为 Qiskit 对象 (委托给 Compiler)"""
        return Compiler.to_qiskit(self)

    @classmethod
    def from_qiskit(cls, qiskit_qc):
        """从 Qiskit 对象重建 (委托给 Compiler)"""
        return Compiler.from_qiskit(qiskit_qc, cls)

    def transpile(self, basis_gates=['u3', 'cx'], optimization_level=1) -> 'QuantumCircuit':
        """
        核心编译功能：
        将当前电路编译为指定基组的电路。
        返回一个新的、编译后的 QuantumCircuit 对象（我们自己的类）。
        """
        return Compiler.transpile(
            self, 
            optimization_level=optimization_level, 
            basis_gates=basis_gates
        )

    def to_qasm(self) -> str:
        """生成 QASM 字符串"""
        return Compiler.to_qasm(self)

    @classmethod
    def from_qasm(cls, qasm_str: str):
        """从 QASM 字符串创建电路"""
        return Compiler.from_qasm(qasm_str, cls)

    # ==========================================
    #           8. 电路统计与分析
    # ==========================================

    def stats(self):
        """
        统计当前电路特征
        直接使用 Compiler 的提取逻辑
        """
        return Compiler.get_circuit_info(self)

    def __repr__(self):
        return f"<QuantumCircuit: qubits={self.n_qubits}, instructions={len(self.instructions)}>"
    
    def draw(self, output='text', **kwargs):
        """
        绘制当前电路
        :param output: 'text' (默认控制台字符画) 或 'mpl' (Matplotlib)
        """
        return Compiler.draw(self, output=output, **kwargs)
    


    # ==========================================
# 附：Backend 自带编译与分析工具包 (CompilerToolkit)
# ==========================================
from typing import Dict, List, Optional, Any, Union
import numpy as np
from collections import Counter
from .core import QuantumCircuit

class Compiler:
    """
    内置编译器工具包：提供原生的编译（Transpile）、线路转换与分析功能。
    """
    @staticmethod
    def _check_qiskit():
        try:
            import qiskit
            return qiskit
        except ImportError:
            raise ImportError("Compiler functions require 'qiskit' installed.")

    @classmethod
    def to_qiskit(cls, circuit: 'QuantumCircuit') -> 'qiskit.QuantumCircuit':
        qiskit = cls._check_qiskit()
        from qiskit import QuantumCircuit as QiskitQC
        import qiskit.circuit.library as qlib

        qc_qiskit = QiskitQC(circuit.n_qubits)
        if circuit.name:
            qc_qiskit.name = circuit.name

        # 门映射表
        GATE_MAP = {
            'id': qlib.IGate, 'x': qlib.XGate, 'y': qlib.YGate, 'z': qlib.ZGate,
            'h': qlib.HGate, 's': qlib.SGate, 'sdg': qlib.SdgGate,
            't': qlib.TGate, 'tdg': qlib.TdgGate, 'sx': qlib.SXGate,
            'swap': qlib.SwapGate, 'iswap': qlib.iSwapGate, 'dcx': qlib.DCXGate, 'ecr': qlib.ECRGate,
            'cnot': qlib.CXGate, 'cx': qlib.CXGate, 'cy': qlib.CYGate, 'cz': qlib.CZGate, 'ch': qlib.CHGate,
            'ccx': qlib.CCXGate, 'cswap': qlib.CSwapGate,
            'rx': qlib.RXGate, 'ry': qlib.RYGate, 'rz': qlib.RZGate,
            'phase': qlib.PhaseGate, 'u3': qlib.U3Gate,
            'cp': qlib.CPhaseGate, 'cphase': qlib.CPhaseGate,
            'crx': qlib.CRXGate, 'cry': qlib.CRYGate, 'crz': qlib.CRZGate,
            'rxx': qlib.RXXGate, 'ryy': qlib.RYYGate, 'rzz': qlib.RZZGate,
            'mcx': qlib.MCXGate
        }

        for inst in circuit.instructions:
            # 1. 处理子电路
            if inst.circuit:
                sub_qiskit = cls.to_qiskit(inst.circuit)
                qc_qiskit.append(sub_qiskit.to_instruction(), inst.qubits)
                continue

            # 2. 处理矩阵 (Unitary)
            if inst.matrix is not None:
                qc_qiskit.unitary(inst.matrix, inst.qubits, label=inst.name or 'Unitary')
                continue
            
            # 3. 处理全局相位
            if inst.name == 'GlobalPhase':
                qc_qiskit.global_phase += inst.params[0]
                continue

            # 4. 处理标准门
            name_lower = inst.name.lower()
            gate_cls = GATE_MAP.get(name_lower)

            if gate_cls:
                if inst.params:
                    gate_obj = gate_cls(*inst.params)
                else:
                    if name_lower == 'mcx':
                        num_ctrl = len(inst.qubits) - 1
                        gate_obj = qlib.MCXGate(num_ctrl)
                    else:
                        gate_obj = gate_cls()
            else:
                print(f"Warning: Gate '{inst.name}' not directly mapped to Qiskit. Skipping.")
                continue

            # 5. 处理控制位 (Control Modifier)
            if inst.control_qubits:
                num_ctrl = len(inst.control_qubits)
                # Qiskit ctrl_state 是字符串，且顺序相反
                ctrl_state = "".join(str(v) for v in reversed(inst.control_values))
                gate_obj = gate_obj.control(num_ctrl, ctrl_state=ctrl_state)
                full_qubits = inst.control_qubits + inst.qubits
                qc_qiskit.append(gate_obj, full_qubits)
            else:
                qc_qiskit.append(gate_obj, inst.qubits)

        return qc_qiskit

    @classmethod
    def from_qiskit(cls, qiskit_qc, circuit_cls) -> 'QuantumCircuit':
        cls._check_qiskit()
        try:
            from qiskit.circuit import ControlledGate
        except ImportError:
            pass 

        qc = circuit_cls(qiskit_qc.num_qubits, name=qiskit_qc.name)

        for instruction in qiskit_qc.data:
            op = instruction.operation
            # 获取 qubit 索引
            qubits_indices = [qiskit_qc.find_bit(q).index for q in instruction.qubits]
            params = [float(p) for p in op.params] if op.params else []
            
            # 处理 Unitary
            if op.name == 'unitary':
                qc.unitary(op.to_matrix(), qubits_indices, check_unitary=False)
                continue

            control_qubits = []
            control_values = []
            target_qubits = qubits_indices
            base_name = op.name
            
            # 处理控制门
            if isinstance(op, ControlledGate):
                num_ctrl = op.num_ctrl_qubits
                control_qubits = qubits_indices[:num_ctrl]
                target_qubits = qubits_indices[num_ctrl:]
                
                ctrl_state = op.ctrl_state
                fmt = f"{{0:0{num_ctrl}b}}"
                ctrl_str = fmt.format(ctrl_state)
                control_values = [int(x) for x in reversed(ctrl_str)]
                base_name = op.base_gate.name

            # 映射名称差异
            map_name = base_name
            if base_name == 'cx': map_name = 'CNOT'
            elif base_name == 'cp': map_name = 'CPhase'
            elif base_name == 'p': map_name = 'Phase'
            
            inst = qc.append(map_name, target_qubits, params=params)
            
            if control_qubits:
                inst.control(control_qubits, control_values)
                
        return qc

    @classmethod
    def transpile(cls, circuit: 'QuantumCircuit', optimization_level=1, basis_gates=None) -> 'QuantumCircuit':
        """
        编译/转译功能：
        将当前电路转换为原生门形式（或指定基组）。
        """
        qiskit = cls._check_qiskit()
        from qiskit import transpile as q_transpile

        # 1. 转换为 Qiskit 对象
        qc_qiskit = cls.to_qiskit(circuit)
        
        # 2. 调用 Qiskit 编译器
        if basis_gates is None:
            basis_gates = ['u3', 'cx'] 
            
        qc_transpiled = q_transpile(
            qc_qiskit, 
            optimization_level=optimization_level,
            basis_gates=basis_gates
        )
        
        # 3. 转换回我们自己的类
        return cls.from_qiskit(qc_transpiled, type(circuit))

    @classmethod
    def to_qasm(cls, circuit: 'QuantumCircuit') -> str:
        """生成 QASM 字符串"""
        cls._check_qiskit()
        import qiskit.qasm2
        qc_qiskit = cls.to_qiskit(circuit)
        # 使用 Qiskit 的接口生成标准 OpenQASM 2.0/3.0
        try:
            return qiskit.qasm2.dumps(qc_qiskit)
        except (ImportError, AttributeError):
            return qc_qiskit.qasm()

    @classmethod
    def from_qasm(cls, qasm_str: str, circuit_cls) -> 'QuantumCircuit':
        """从 QASM 字符串加载电路"""
        cls._check_qiskit()
        import qiskit.qasm2
        try:
            qiskit_qc = qiskit.qasm2.loads(qasm_str)
        except Exception:
            from qiskit import QuantumCircuit as QiskitQC
            qiskit_qc = QiskitQC.from_qasm_str(qasm_str)
            
        return cls.from_qiskit(qiskit_qc, circuit_cls)

    @staticmethod
    def get_circuit_info(circuit: 'QuantumCircuit') -> Dict[str, Any]:
        """
        提取电路信息
        """
        info = {
            "n_qubits": circuit.n_qubits,
            "n_instructions": len(circuit.instructions),
            "gate_counts": Counter(),
            "multi_qubit_gates": 0,
            "depth_approx": 0,  # 深度估计
            "parameterized": False
        }
        
        # 用于计算深度的简单数组 (记录每个 qubit 当前的时间步)
        qubit_timesteps = [0] * circuit.n_qubits

        for inst in circuit.instructions:
            name = inst.name
            info["gate_counts"][name] += 1
            
            # 涉及的所有比特 (目标 + 控制)
            all_qubits = inst.qubits + inst.control_qubits
            
            if len(all_qubits) > 1:
                info["multi_qubit_gates"] += 1
            
            if inst.params:
                info["parameterized"] = True
                
            # 计算深度 (当前门必须在所有涉及 qubit 的前置门之后执行)
            if all_qubits:
                current_depth = max(qubit_timesteps[q] for q in all_qubits) + 1
                for q in all_qubits:
                    qubit_timesteps[q] = current_depth

        info["depth_approx"] = max(qubit_timesteps) if qubit_timesteps else 0
        return info
    
    @classmethod
    def draw(cls, circuit: 'QuantumCircuit', output='text', **kwargs):
        """
        绘制电路图
        :param output: 输出格式，支持 'text' (字符画), 'mpl' (Matplotlib图片), 'latex' 等
        """
        cls._check_qiskit()
        qc_qiskit = cls.to_qiskit(circuit)
        return qc_qiskit.draw(output=output, **kwargs)

