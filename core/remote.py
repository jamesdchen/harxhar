"""Backward-compatible remote utilities delegating to the hpc package.

Preserves the harxhar-specific defaults (Hoffman2, jamesdc1) and module-level
config variables so existing callers don't need to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import subprocess

__all__ = [
    "REMOTE_HOST",
    "REMOTE_USER",
    "REMOTE_REPO",
    "ssh_run",
    "rsync_push",
    "rsync_pull_results",
]

import os
from pathlib import Path

from hpc.remote import rsync_pull
from hpc.remote import rsync_push as _rsync_push
from hpc.remote import ssh_run as _ssh_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REMOTE_HOST = os.environ.get("HPC_HOST", "hoffman2.idre.ucla.edu")
REMOTE_USER = os.environ.get("HPC_USER", "jamesdc1")
REMOTE_REPO = os.environ.get("HPC_REPO", "/u/home/j/jamesdc1/project-cucuringu/harxhar")

RSYNC_EXCLUDES = [
    ".git/",
    "results/",
    "__pycache__/",
    "*.pyc",
    ".mypy_cache/",
    "all30min/",
    ".claude/",
]


def ssh_run(cmd: str, *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run *cmd* on the cluster via SSH."""
    return _ssh_run(cmd, host=REMOTE_HOST, user=REMOTE_USER, capture=capture)


def rsync_push() -> subprocess.CompletedProcess[str]:
    """Sync local code to the cluster."""
    return _rsync_push(
        host=REMOTE_HOST,
        user=REMOTE_USER,
        remote_path=REMOTE_REPO,
        local_path=PROJECT_ROOT,
        exclude=RSYNC_EXCLUDES,
    )


def rsync_pull_results(
    remote_dir: str = "results/",
    local_dir: str = "results/",
) -> subprocess.CompletedProcess[str]:
    """Pull summary CSVs and metadata from the cluster."""
    dst_path = PROJECT_ROOT / local_dir.rstrip("/\\")
    return rsync_pull(
        host=REMOTE_HOST,
        user=REMOTE_USER,
        remote_path=REMOTE_REPO,
        remote_subdir=remote_dir,
        local_dir=dst_path,
        include=[
            "*_summary*.csv",
            "metadata.json",
            "config.txt",
            "lifecycle.jsonl",
        ],
    )
