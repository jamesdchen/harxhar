"""Backward-compatible re-export of HPC backends from the claude-hpc package."""

from __future__ import annotations

__all__ = [
    "HPCBackend",
    "DryRunBackend",
    "get_backend",
    "register",
    "PROJECT_ROOT",
    "resolve_template",
    "build_stage_env",
]

from pathlib import Path

from hpc import get_template_path, load_clusters_config, load_project_config
from hpc.backends import HPCBackend, register
from hpc.backends import get_backend as _hpc_get_backend
from hpc.backends.dry_run import DryRunBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_template(scheduler: str, template: str) -> str:
    return str(get_template_path(scheduler, template))


def build_stage_env(cluster_name: str, stage_name: str) -> dict[str, str]:
    clusters = load_clusters_config()
    project = load_project_config()

    cluster = clusters[cluster_name]
    stage = project["stages"][stage_name]
    env_group = stage["env_group"]
    env = project["cluster_envs"][cluster_name][env_group]

    result: dict[str, str] = {
        "MODULES": env.get("modules", ""),
        "REPO_DIR": project["remote_path"],
        "EXECUTOR": stage["executor"],
    }

    conda_env = env.get("conda_env")
    if conda_env is not None:
        result["CONDA_SOURCE"] = cluster["conda_source"]
        result["CONDA_ENV"] = conda_env

    return result


def get_backend(name: str = "slurm", **kwargs: object) -> HPCBackend:
    """Instantiate a backend by name, injecting harxhar SSH config for sge-remote."""
    if name == "sge-remote" and "ssh_run" not in kwargs:
        from core.remote import REMOTE_REPO, ssh_run

        kwargs.setdefault("ssh_run", ssh_run)
        kwargs.setdefault("remote_repo", REMOTE_REPO)
    return _hpc_get_backend(name, **kwargs)
