"""Backward-compatible re-export — backends now live in core.backends."""

from core.backends import (  # noqa: F401
    PROJECT_ROOT,
    DryRunBackend,
    HPCBackend,
    get_backend,
    register,
)

__all__ = [
    "HPCBackend",
    "DryRunBackend",
    "get_backend",
    "register",
    "PROJECT_ROOT",
]
