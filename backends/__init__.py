"""
Backend exports and registry for HYQ-ALG-LIB.

This module exposes both the original backends and optional extended backends.
It also provides a small registry for dynamic backend selection.
"""

from __future__ import annotations

import os
from typing import Dict, Type


try:
    from .Qiskit import QiskitBackend
except Exception:  # pragma: no cover
    QiskitBackend = None  # type: ignore

try:
    from .Pennylane import PennyLaneBackend
except Exception:  # pragma: no cover
    PennyLaneBackend = None  # type: ignore

try:
    from .Tensorcircuit import TensorCircuitBackend
except Exception:  # pragma: no cover
    TensorCircuitBackend = None  # type: ignore

try:
    from .Cirq import CirqBackend
except Exception:  # pragma: no cover
    CirqBackend = None  # type: ignore

try:
    from .Qulacs import QulacsBackend
except Exception:  # pragma: no cover
    QulacsBackend = None  # type: ignore

try:
    from .Qutip import QutipBackend
except Exception:  # pragma: no cover
    QutipBackend = None  # type: ignore


_BACKEND_CLASSES: Dict[str, Type] = {}

if QiskitBackend is not None:
    _BACKEND_CLASSES["qiskit"] = QiskitBackend

if PennyLaneBackend is not None:
    _BACKEND_CLASSES["pennylane"] = PennyLaneBackend

if TensorCircuitBackend is not None:
    _BACKEND_CLASSES["tensorcircuit"] = TensorCircuitBackend

if CirqBackend is not None:
    _BACKEND_CLASSES["cirq"] = CirqBackend

if QulacsBackend is not None:
    _BACKEND_CLASSES["qulacs"] = QulacsBackend

if QutipBackend is not None:
    _BACKEND_CLASSES["qutip"] = QutipBackend


_ACTIVE_BACKEND = None


def available_backends():
    """Return the names of currently importable backends."""
    return sorted(_BACKEND_CLASSES.keys())


def register_backend(name: str, backend_cls: Type):
    """Register a backend class under a lowercase name."""
    _BACKEND_CLASSES[name.lower()] = backend_cls


def set_backend(name: str, **kwargs):
    """Instantiate and set the active backend."""
    global _ACTIVE_BACKEND

    key = name.lower()
    if key not in _BACKEND_CLASSES:
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {available_backends()}"
        )

    _ACTIVE_BACKEND = _BACKEND_CLASSES[key](**kwargs)
    os.environ["HYQ_BACKEND"] = key
    return _ACTIVE_BACKEND


def get_backend(default: str = "qiskit", **kwargs):
    """Return the active backend, creating one from HYQ_BACKEND if needed."""
    global _ACTIVE_BACKEND

    if _ACTIVE_BACKEND is None:
        name = os.environ.get("HYQ_BACKEND", default)
        return set_backend(name, **kwargs)

    return _ACTIVE_BACKEND


__all__ = [
    "QiskitBackend",
    "PennyLaneBackend",
    "TensorCircuitBackend",
    "CirqBackend",
    "QulacsBackend",
    "QutipBackend",
    "available_backends",
    "register_backend",
    "set_backend",
    "get_backend",
]