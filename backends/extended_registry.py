"""
Backward-compatible registry imports.

Prefer importing these functions directly from backends:
    from backends import set_backend, get_backend
"""

from . import available_backends, register_backend, set_backend, get_backend

__all__ = [
    "available_backends",
    "register_backend",
    "set_backend",
    "get_backend",
]