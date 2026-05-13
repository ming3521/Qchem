# HyQ：混合量子算法工具包
HyQ 是一个轻量级 Python 工具包，用于构建与后端无关的量子线路并运行混合量子算法，例如 VQE、VarQITE、SA‑QITE、SS‑QITE 和 RITE。它提供：

- 用于构建线路的统一 `QuantumCircuit` 抽象。
- 可插拔的后端接口，支持两种执行模式：
    - 采样模式：返回测量计数，类似于硬件或基于采样的模拟器。
    - 状态向量模式：返回精确状态向量，适用于模拟器和可微分算法。
- 针对 Qiskit、PennyLane 和 TensorCircuit 的后端实现。
- 可选化学工具，使用 OpenFermion 和 Psi4 生成分子哈密顿量。

> 注意：在已上传的源文件中，后端和求解器的基类都命名为 `base.py`。在仓库中请将它们放在不同的包中，例如 `backends/base.p`y 和 `solvers/base.py`。

## 推荐的仓库结构

```text
.
├── backends/
│   ├── __init__.py
│   ├── base.py
│   ├── core.py
│   ├── Qiskit.py
│   ├── Pennylane.py
│   └── Tensorcircuit.py
├── chemistry/
│   ├── __init__.py
│   ├── hamiltonian.py
│   ├── molecule.py
│   └── psi4_driver.py
├── solvers/
│   ├── __init__.py
│   ├── base.py
│   ├── vqe.py
│   ├── var_qite.py
│   ├── sa_qite.py
│   ├── ss_qite.py
│   └── rite.py
├── examples/
│   ├── example_hamiltonian.ipynb
│   ├── example_solvers.ipynb
│   └── example_compiler.ipynb
├── tutorials/
│   ├── tutorial_comparison.ipynb
│   ├── tutorial_rmite.ipynb
│   ├── tutorial_saqite.ipynb
│   ├── tutorial_rmite.ipynb
│   └── tutorial_varqite.ipynb
│   └── tutorial_vqe.ipynb
├── requirements.txt
├── requirements-qiskit.txt
├── requirements-pennylane.txt
├── requirements-tensorcircuit.txt
├── requirements-chemistry.txt
└── README.md
```

## 安装

首先创建并激活虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -U pip
```

安装核心依赖：
```bash
pip install -r requirements.txt
```

根据您的使用场景安装至少一个后端。

### 1. 只需要基于采样的运行（类似在硬件上运行）

推荐后端：Qiskit。

```bash
pip install -r requirements-qiskit.txt
# Linux / macOS
export HYQ_BACKEND=qiskit
# Windows (CMD)
set HYQ_BACKEND=qiskit
# Windows (PowerShell)
$env:HYQ_BACKEND="qiskit"
```

Qiskit 模式支持 `run_sampling(circuit, shots, measure_qubits)`，并返回计数字典，例如 `{"00": 512, "11": 512}`。

### 2. 需要可微分的状态向量模拟（用于 VQE / VarQITE）

推荐后端：TensorCircuit 或 PennyLane。

TensorCircuit：

```bash
pip install -r requirements-tensorcircuit.txt
# Linux / macOS
export HYQ_BACKEND=tensorcircuit
# Windows (CMD)
set HYQ_BACKEND=tensorcircuit
# Windows (PowerShell)
$env:HYQ_BACKEND="tensorcircuit"
```

PennyLane：

```bash
pip install -r requirements-pennylane.txt
# Linux / macOS
export HYQ_BACKEND=pennylane
# Windows (CMD)
set HYQ_BACKEND=pennylane
# Windows (PowerShell)
$env:HYQ_BACKEND="pennylane"
```

这些后端通过 `get_statevector(circuit)` 支持 PyTorch 自动微分，是依赖梯度的算法的首选。

### 3. 需要 Qiskit 的线路转换、编译、QASM 或绘图

安装 Qiskit：

```bash
pip install -r requirements-qiskit.txt
# Linux / macOS
export HYQ_BACKEND=qiskit
# Windows (CMD)
set HYQ_BACKEND=qiskit
# Windows (PowerShell)
$env:HYQ_BACKEND="qiskit"
```

公共 circuit 对象提供了 `to_qiskit()`、`from_qiskit()`、`transpile()`、`to_qasm()`、`from_qasm()`、`stats()`和 `draw()`等辅助方法。

### 4. 需要分子哈密顿量

安装化学依赖：

```bash
pip install -r requirements-chemistry.txt
```

Psi4 可能需要根据您的平台单独通过 conda 或系统级方式安装。Python 包 `openfermionpsi4` 内部调用 Psi4，因此在使用 `chemistry/psi4_driver.py` 之前，请确保您的环境中已有 `psi4`。

```bash
conda install -c psi4 psi4
```

## 后端能力指南

| 后端 | 采样模式 | 状态向量模式 | 自动微分友好 | 推荐用途 |
|---|---:|---:|---:|---|
| Qiskit | 支持 | 支持 | 不支持 | 基于采样的模拟、QASM、编译、硬件风格的工作流 |
| PennyLane | 支持 | 支持 | 支持 | 可微分模拟和算法原型开发 |
| TensorCircuit | 支持 | 支持 | 支持 | 快速的可微分状态向量工作流 |

## 最小使用示例

```python
import torch
from backends.core import QuantumCircuit
from backends.Tensorcircuit import TensorCircuitBackend

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

backend = TensorCircuitBackend()
state = backend.get_statevector(qc)
print(state)

counts = backend.run_sampling(qc, shots=1024, measure_qubits=[0, 1])
print(counts)
```
## 使用求解器

```python
import torch
from backends.core import QuantumCircuit
from backends.Tensorcircuit import TensorCircuitBackend
from solvers.vqe import VQESolver

backend = TensorCircuitBackend()
solver = VQESolver(backend)

def ansatz(params):
    qc = QuantumCircuit(2)
    qc.ry(0, params[0])
    qc.cx(0, 1)
    return qc

hamiltonian = [(-1.0, "ZI"), (-1.0, "IZ"), (0.5, "XX")]
init_params = torch.tensor([0.1], dtype=torch.float64)
energy, params, history = solver.solve(ansatz, init_params, hamiltonian, steps=100, lr=0.1)
```

## 生成分子哈密顿量

```python
from chemistry.hamiltonian import Hamiltonian

symbols = ["H", "H"]
geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

ham = Hamiltonian(symbols, geometry, charge=0, multiplicity=1, basis="sto-3g")
terms, n_qubits, n_electrons = ham.get_processed_hamiltonian(
    n_active_electrons=2,
    n_active_orbitals=2,
    mapping="jw",
    reverse_endian=False,
)
```



| 后端            | 采样 `run_sampling` | 状态向量 `get_statevector` |         是否支持自动微分 |             是否原生执行 HYQ gate | 评价                        |
| ------------- | ----------------: | ---------------------: | ---------------: | --------------------------: | ------------------------- |
| Qiskit        |                支持 |                     支持 | 不支持 PyTorch 自动微分 |            通过 `to_qiskit()` | 老后端，适合画图/采样/数值态矢          |
| PennyLane     |                支持 |                     支持 |    支持 PyTorch 接口 |                 手写 gate map | 老后端，适合可微分，但 gate 覆盖有限     |
| TensorCircuit |                支持 |                     支持 |       支持 PyTorch | 手写 gate map + fallback 小写方法 | 老后端，适合 VarQITE/VQE 梯度     |
| Cirq          |                支持 |                     支持 | 不支持 PyTorch 自动微分 |     多数门原生，失败 fallback dense | 新增，可用但有一个 sampling 潜在 bug |
| Qulacs        |                支持 |                     支持 | 不支持 PyTorch 自动微分 |          用 DenseMatrix 统一执行 | 新增，整体最稳                   |
| Qutip         |                支持 |                     支持 | 不支持 PyTorch 自动微分 |  实际用 shared dense simulator | 新增，更像 QuTiP 入口包装          |


