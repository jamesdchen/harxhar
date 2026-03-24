"""
Pluggable HPC backend system.

Provides an abstract interface for job submission so any project
can target any scheduler (SLURM, SGE, PBS, ...) without changing
the core submission logic.

Usage:
    from core.backends import get_backend
    backend = get_backend("slurm", script="path/to/job.slurm")
    backend.submit_array(job_name, total_chunks, tasks_per_array, job_env)
"""

from __future__ import annotations

__all__ = [
    "HPCBackend",
    "DryRunBackend",
    "get_backend",
    "register",
    "PROJECT_ROOT",
]

import abc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class HPCBackend(abc.ABC):
    """Minimal interface for HPC job submission backends."""

    @abc.abstractmethod
    def submit_array(
        self,
        job_name: str,
        total_chunks: int,
        tasks_per_array: int,
        job_env: dict[str, str],
    ) -> None:
        """Submit an array job.  Each task receives its chunk ID via the scheduler."""
        ...


_REGISTRY: dict[str, type[HPCBackend]] = {}


def register(name: str):
    """Decorator to register a backend class."""

    def decorator(cls: type[HPCBackend]) -> type[HPCBackend]:
        _REGISTRY[name] = cls
        return cls

    return decorator


@register("dry-run")
class DryRunBackend(HPCBackend):
    """Print what would be submitted without actually running anything."""

    def __init__(self, **kwargs):
        pass  # Accept and ignore backend-specific kwargs (e.g. script)

    def submit_array(self, job_name, total_chunks, tasks_per_array, job_env):
        result_dir = job_env.get("RESULT_DIR", "?")
        print(f"  [DRY RUN] Job: {job_name}")
        print(f"            Chunks: 1-{total_chunks} (batches of {tasks_per_array})")
        print(f"            Output: {result_dir}")
        for k in ("MODEL_TYPE", "EXPERIMENT"):
            if k in job_env and job_env[k]:
                print(f"            {k}: {job_env[k]}")
        extra_args = job_env.get("EXTRA_ARGS", "")
        if extra_args:
            print(f"            Extra args: {extra_args}")
        print()


def get_backend(name: str = "slurm", **kwargs) -> HPCBackend:
    """Instantiate a backend by name.  *kwargs* are forwarded to the constructor."""
    # Lazy imports to populate registry
    from core.backends import sge as _sge  # noqa: F401
    from core.backends import slurm as _slurm  # noqa: F401

    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
