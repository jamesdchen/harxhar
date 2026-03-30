"""Backward-compatible re-export of HPC backends from the claude-hpc package."""

from __future__ import annotations

__all__ = [
    "HPCBackend",
    "DryRunBackend",
    "get_backend",
    "register",
    "PROJECT_ROOT",
]

from pathlib import Path

from hpc.backends import HPCBackend, register
from hpc.backends import get_backend as _hpc_get_backend
from hpc.backends.dry_run import DryRunBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_backend(name: str = "slurm", **kwargs: object) -> HPCBackend:
    """Instantiate a backend by name, injecting harxhar SSH config for sge-remote."""
    if name == "sge-remote" and "ssh_run" not in kwargs:
        from core.remote import REMOTE_REPO, ssh_run

        kwargs.setdefault("ssh_run", ssh_run)
        kwargs.setdefault("remote_repo", REMOTE_REPO)
    return _hpc_get_backend(name, **kwargs)
