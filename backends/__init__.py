"""Extended backend exports for HYQ-ALG-LIB."""

try:
    from .Qiskit import QiskitBackend
except Exception:  # pragma: no cover
    QiskitBackend = None  # type: ignore
try:
    from .Pennylane import PennylaneBackend
except Exception:  # pragma: no cover
    PennylaneBackend = None  # type: ignore
try:
    from .Tensorcircuit import TensorCircuitBackend
except Exception:  # pragma: no cover
    TensorCircuitBackend = None  # type: ignore

from .Cirq import CirqBackend
from .Qulacs import QulacsBackend
from .Qutip import QutipBackend

EXTENDED_BACKENDS = {
    "cirq": CirqBackend,
    "qulacs": QulacsBackend,
    "qutip": QutipBackend,
}

__all__ = [
    "CirqBackend",
    "QulacsBackend",
    "QutipBackend",
    "EXTENDED_BACKENDS",
]
