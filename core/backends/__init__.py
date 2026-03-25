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
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class HPCBackend(abc.ABC):
    """Minimal interface for HPC job submission backends.

    Subclasses implement ``_build_command`` to construct the scheduler-specific
    command; the chunking loop and subprocess execution are handled here.
    """

    log_dir: str  # subclasses must set this

    @abc.abstractmethod
    def _build_command(
        self,
        task_range: str,
        job_name: str,
        job_env: dict[str, str],
    ) -> list[str]:
        """Return the scheduler command for the given task range."""
        ...

    def submit_array(
        self,
        job_name: str,
        total_chunks: int,
        tasks_per_array: int,
        job_env: dict[str, str],
    ) -> None:
        """Submit an array job in batches of *tasks_per_array*."""
        os.makedirs(self.log_dir, exist_ok=True)

        start_task = 1
        while start_task <= total_chunks:
            end_task = min(start_task + tasks_per_array - 1, total_chunks)
            task_range = f"{start_task}-{end_task}"
            cmd = self._build_command(task_range, job_name, job_env)
            result = subprocess.run(
                cmd,
                env=job_env,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr_msg = result.stderr.strip() if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"Job submission failed (exit {result.returncode}) for array {task_range}:\n"
                    f"  command: {' '.join(cmd)}\n"
                    f"  stderr:  {stderr_msg}"
                )
            start_task = end_task + 1

    def submit_array_tracked(
        self,
        job_name: str,
        total_chunks: int,
        tasks_per_array: int,
        job_env: dict[str, str],
    ) -> list[tuple[str, str]]:
        """Like submit_array but returns (task_range, job_id) pairs."""
        os.makedirs(self.log_dir, exist_ok=True)
        submissions: list[tuple[str, str]] = []

        start_task = 1
        while start_task <= total_chunks:
            end_task = min(start_task + tasks_per_array - 1, total_chunks)
            task_range = f"{start_task}-{end_task}"
            cmd = self._build_command(task_range, job_name, job_env)
            result = subprocess.run(
                cmd,
                env=job_env,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr_msg = result.stderr.strip() if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"Job submission failed (exit {result.returncode}) for array {task_range}:\n"
                    f"  command: {' '.join(cmd)}\n"
                    f"  stderr:  {stderr_msg}"
                )
            match = re.search(r"(\d+)", result.stdout)
            if not match:
                raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout!r}")
            submissions.append((task_range, match.group(1)))
            start_task = end_task + 1

        return submissions


_REGISTRY: dict[str, type[HPCBackend]] = {}


def register(name: str) -> Callable[[type[HPCBackend]], type[HPCBackend]]:
    """Decorator to register a backend class."""

    def decorator(cls: type[HPCBackend]) -> type[HPCBackend]:
        _REGISTRY[name] = cls
        return cls

    return decorator


@register("dry-run")
class DryRunBackend(HPCBackend):
    """Print what would be submitted without actually running anything."""

    def __init__(self, **kwargs: object) -> None:
        self.log_dir = ""  # unused, satisfies base class attribute
        pass  # Accept and ignore backend-specific kwargs (e.g. script)

    def _build_command(self, task_range: str, job_name: str, job_env: dict[str, str]) -> list[str]:
        return []  # Not used — submit_array is overridden entirely

    def submit_array(
        self,
        job_name: str,
        total_chunks: int,
        tasks_per_array: int,
        job_env: dict[str, str],
    ) -> None:
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


def get_backend(name: str = "slurm", **kwargs: object) -> HPCBackend:
    """Instantiate a backend by name.  *kwargs* are forwarded to the constructor."""
    # Lazy imports to populate registry
    from core.backends import sge as _sge  # noqa: F401
    from core.backends import slurm as _slurm  # noqa: F401

    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
