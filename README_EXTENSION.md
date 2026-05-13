# Qchem extension code for software-copyright code expansion

This directory contains drop-in Python files for the public `ming3521/Qchem`
repository.  The extension follows the existing folder layout:

- `backends/`: adds `CirqBackend`, `QulacsBackend`, `QutipBackend` and shared
  dense simulation utilities.
- `ansatz/`: adds `RyRzAnsatz`, compact pair-excitation / k-UpCCG ansatz, and an
  ADAPT-VQE style adaptive ansatz toolkit.
- `algorithms/`: adds Trotter simulation, phase estimation, classical exact and
  Lanczos eigensolvers, QSE, VQD, MP2/FCI/fidelity/symmetry utilities.

Suggested copy command from the repository root:

```bash
cp -r /path/to/qchem_extension/backends/*.py backends/
cp -r /path/to/qchem_extension/ansatz/*.py ansatz/
cp -r /path/to/qchem_extension/algorithms/*.py algorithms/
```

Optional dependencies:

```bash
pip install cirq qulacs qutip qutip-qip scipy
```

The code is designed so optional backend packages are imported only when those
backends are actually instantiated.
