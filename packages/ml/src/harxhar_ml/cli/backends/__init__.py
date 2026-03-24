"""
Pluggable HPC backend system.

Provides an abstract interface for job submission so the framework
can target any scheduler (SLURM, SGE, PBS, …) without changing
the core submission logic.

Usage:
    from harxhar_ml.cli.backends import get_backend
    backend = get_backend("slurm")          # or "sge", etc.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


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
        """Submit an array job. Each task runs src.cli.executor with a chunk ID."""
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

    def submit_array(self, job_name, total_chunks, tasks_per_array, job_env):
        result_dir = job_env.get("RESULT_DIR", "?")
        model_type = job_env.get("MODEL_TYPE", "?")
        extra_args = job_env.get("EXTRA_ARGS", "")
        exog_cols = job_env.get("EXOG_COLS", "None")

        print(f"  [DRY RUN] Job: {job_name}")
        print(f"            Model: {model_type}")
        print(f"            Chunks: 1-{total_chunks} (batches of {tasks_per_array})")
        print(f"            Output: {result_dir}")
        if exog_cols != "None":
            n_vars = len(exog_cols.split("|"))
            print(f"            Exog vars: {n_vars}")
        if extra_args:
            print(f"            Extra args: {extra_args}")
        print()


def get_backend(name: str = "slurm", **kwargs) -> HPCBackend:
    """Instantiate a backend by name.  *kwargs* are forwarded to the constructor."""
    # Lazy imports to populate registry
    from harxhar_ml.cli.backends import sge as _sge  # noqa: F401
    from harxhar_ml.cli.backends import slurm as _slurm  # noqa: F401

    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
