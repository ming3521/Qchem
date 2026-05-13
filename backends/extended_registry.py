"""Small backend registry for projects that want dynamic backend selection."""

from __future__ import annotations

import os
from typing import Any, Dict, Type

from .Cirq import CirqBackend
from .Qulacs import QulacsBackend
from .Qutip import QutipBackend

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

_BACKEND_CLASSES: Dict[str, Type] = {
    "cirq": CirqBackend,
    "qulacs": QulacsBackend,
    "qutip": QutipBackend,
}
if QiskitBackend is not None:
    _BACKEND_CLASSES["qiskit"] = QiskitBackend
if PennylaneBackend is not None:
    _BACKEND_CLASSES["pennylane"] = PennylaneBackend
if TensorCircuitBackend is not None:
    _BACKEND_CLASSES["tensorcircuit"] = TensorCircuitBackend

_ACTIVE_BACKEND = None


def available_backends():
    return sorted(_BACKEND_CLASSES.keys())


def register_backend(name: str, backend_cls: Type):
    _BACKEND_CLASSES[name.lower()] = backend_cls


def set_backend(name: str, **kwargs):
    global _ACTIVE_BACKEND
    key = name.lower()
    if key not in _BACKEND_CLASSES:
        raise ValueError(f"Unknown backend '{name}'. Available: {available_backends()}")
    _ACTIVE_BACKEND = _BACKEND_CLASSES[key](**kwargs)
    os.environ["HYQ_BACKEND"] = key
    return _ACTIVE_BACKEND


def get_backend(default: str = "qiskit", **kwargs):
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        name = os.environ.get("HYQ_BACKEND", default)
        return set_backend(name, **kwargs)
    return _ACTIVE_BACKEND
