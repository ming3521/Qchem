from chemistry.molecule import Molecule

class Hamiltonian:
    """
    哈密顿量生成器类。
    负责从分子结构生成可直接用于量子算法后端的哈密顿量列表。
    自动处理活性空间投影、格式转换及大端序翻转。
    """
    def __init__(self, symbols, geometry, charge=0, multiplicity=1, basis='sto-3g'):
        """
        初始化分子基本信息
        Args:
            symbols (list): 元素符号列表，如 ["H", "H"]
            geometry (list): 坐标列表
            charge (int): 电荷
            multiplicity (int): 自旋多重度
            basis (str): 基组
        """
        self.molecule = Molecule(
            symbols=symbols,
            geometry=geometry,
            basis=basis,
            multiplicity=multiplicity,
            charge=charge
        )
        # 初始化时立即执行 Psi4 计算
        print(">>> [Hamiltonian] 正在调用 Psi4 计算分子积分...")
        self.molecule.run()

    def get_processed_hamiltonian(self, n_active_electrons=None, n_active_orbitals=None, mapping="jw", reverse_endian=False):
        """
        生成经过处理的哈密顿量（可配置是否颠倒大端序）。
        
        Args:
            n_active_electrons (int): 活性空间电子数
            n_active_orbitals (int): 活性空间轨道数
            mapping (str): 映射方法 ("jw" 或 "bk")
            reverse_endian (bool): 是否翻转 Pauli 字符串顺序 (Big-Endian -> Little-Endian)。
                                   默认为 True (适配 TensorCircuit/Qiskit 等后端)。
                                   若需匹配 OpenFermion 原始顺序，请设为 False。
            
        Returns:
            tuple: (hamiltonian_list, n_qubits, n_electrons)
                - hamiltonian_list: [(coeff, "ZI..."), ...] 格式
                - n_qubits: 总比特数
                - n_electrons: 考虑活性空间后的有效电子数
        """
        # 1. 设置活性空间
        if n_active_electrons is not None and n_active_orbitals is not None:
            print(f">>> [Hamiltonian] 设置活性空间: {n_active_electrons}e, {n_active_orbitals}orb")
            self.molecule.set_active_space(n_active_electrons, n_active_orbitals)
            # 活性空间下的总比特数 = 2 * 空间轨道数
            n_qubits = 2 * n_active_orbitals
            final_n_electrons = n_active_electrons # 算法关注的是活性电子
        else:
            # 全活性空间
            n_qubits = self.molecule.data.n_qubits
            final_n_electrons = self.molecule.data.n_electrons

        # 2. 获取原始 OpenFermion QubitOperator
        qubit_op = self.molecule.qubit_hamiltonian(mapping)
        print(f">>> [Hamiltonian] 原始算符项数: {len(qubit_op.terms)}")

        # 3. 转换为列表格式
        processed_list = []
        
        for term, coeff in qubit_op.terms.items():
            # 构建原始顺序的 Pauli 串 (q0, q1, ... qN)
            pauli_chars = ['I'] * n_qubits
            for qubit_idx, pauli_op in term:
                if qubit_idx < n_qubits:
                    pauli_chars[qubit_idx] = pauli_op
                else:
                    raise ValueError(f"Qubit index {qubit_idx} out of range {n_qubits}")
            
            raw_string = "".join(pauli_chars)
            
            # 处理端序 (Endianness) 
            if reverse_endian:
                # 例如 "IZ" (q0=I, q1=Z) -> "ZI" (适配 TensorCircuit/Qiskit 常见顺序)
                processed_string = raw_string[::-1]
            else:
                # 保持原始顺序 (适配 OpenFermion )
                processed_string = raw_string
            
            # 提取实部系数
            val = coeff.real if isinstance(coeff, complex) else coeff
            processed_list.append((val, processed_string))
            
        return processed_list, n_qubits, final_n_electrons