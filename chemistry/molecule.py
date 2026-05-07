import sys
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.transforms import bravyi_kitaev_code
from .psi4_driver import Psi4Driver
import numpy as np

class Molecule:
    def __init__(self, symbols, geometry, basis='sto-3g', multiplicity=1, charge=0):
        self.symbols = symbols
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge

        self.occupied_indices = None
        self.active_indices = None
        
        self.data = None
        

    def set_active_space(self, n_active_electrons, n_active_orbitals):
        n_core_orbitals = (self.data.n_electrons - n_active_electrons) // 2
        self.occupied_indices = list(range(n_core_orbitals))
        self.active_indices = list(range(n_core_orbitals, n_core_orbitals + n_active_orbitals))

    def run(self, filename=None, run_fci=False):
        driver = Psi4Driver(self.symbols, self.geometry, self.basis, self.multiplicity, self.charge)
        self.data = driver.run(filename, run_fci)
    
    def fermionic_hamiltonian(self, n_active_electrons=None, n_active_orbitals=None):
        if n_active_electrons is not None and n_active_orbitals is not None:
            self.set_active_space(n_active_electrons, n_active_orbitals)
        fermionic_hamiltonian = self.data.get_molecular_hamiltonian(
            occupied_indices=self.occupied_indices,
            active_indices=self.active_indices
        )
        return fermionic_hamiltonian

    def qubit_hamiltonian(self, mapping="jw", custom_mapping=None):
        if custom_mapping is not None:
            qubit_hamiltonian = custom_mapping(self.fermionic_hamiltonian())
        else:
            if mapping == "jw":
                qubit_hamiltonian = jordan_wigner(self.fermionic_hamiltonian())
            elif mapping == "bk":
                qubit_hamiltonian = bravyi_kitaev(self.fermionic_hamiltonian())
            else:
                raise ValueError(f"Invalid mapping: {mapping}")
        return qubit_hamiltonian

    def hf_state(self, mapping="jw"): 
        # 1. 获取活性空间信息
        # 假设你的 self.data 是 MolecularData 对象
        n_qubits = self.data.n_qubits # 这里的比特数已经是活性空间的 2 * n_active_orbitals
        
        # 计算活性电子数 (Active Electrons)
        # 如果没有定义 active_space，则 n_active_electrons = n_electrons
        if self.active_indices is None:
            n_active_electrons = self.data.n_electrons
        else:
            n_active_electrons = len(self.active_indices)
        
        # 2. 物理合法性检查
        if (n_active_electrons + self.multiplicity) % 2 == 0:
            raise ValueError(f"活性电子数与多重度不匹配！")

        # 3. 计算活性空间内的 Alpha 和 Beta
        # 注意：这里的分配是针对活性空间内部的
        n_active_alpha = (n_active_electrons + self.multiplicity - 1) // 2
        n_active_beta = n_active_electrons - n_active_alpha

        # 4. 初始化活性空间的费米子占据向量
        fermion_occupations = np.zeros(n_qubits, dtype=int)
        
        # 在活性轨道内按能量排序填充
        # Alpha 占据活性空间的前几个 Alpha 位置
        for i in range(n_active_alpha):
            if 2 * i < n_qubits:
                fermion_occupations[2 * i] = 1
                
        # Beta 占据活性空间的前几个 Beta 位置
        for i in range(n_active_beta):
            if 2 * i + 1 < n_qubits:
                fermion_occupations[2 * i + 1] = 1

        # 5. 映射逻辑 (保持不变，因为 n_qubits 已是活性比特数)
        if mapping.lower() == 'jw':
            qubit_occupations = fermion_occupations
        elif mapping.lower() == 'bk':
            bk_matrix = bravyi_kitaev_code(n_qubits).encoder.toarray()
            qubit_occupations = (bk_matrix @ fermion_occupations) % 2
        else:
            raise NotImplementedError(f"映射不支持: {mapping}")

        return [i for i, val in enumerate(qubit_occupations) if val == 1]
    