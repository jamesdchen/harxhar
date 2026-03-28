"""SSH and rsync utilities for remote HPC operations.

Provides thin wrappers around ssh/rsync so cluster commands can be
executed from a local machine without paramiko or other dependencies.

Configuration via environment variables:
    HPC_HOST  — cluster hostname  (default: hoffman2.idre.ucla.edu)
    HPC_USER  — cluster username  (default: jamesdc1)
    HPC_REPO  — repo path on cluster (default: /u/project/project-cucuringu/harxhar)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REMOTE_HOST = os.environ.get("HPC_HOST", "hoffman2.idre.ucla.edu")
REMOTE_USER = os.environ.get("HPC_USER", "jamesdc1")
REMOTE_REPO = os.environ.get("HPC_REPO", "/u/project/project-cucuringu/harxhar")

RSYNC_EXCLUDES = [
    ".git/",
    "results/",
    "__pycache__/",
    "*.pyc",
    ".mypy_cache/",
    "all30min/",
    ".claude/",
]


def _target() -> str:
    return f"{REMOTE_USER}@{REMOTE_HOST}"


def ssh_run(cmd: str, *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run *cmd* on the cluster via SSH.

    Parameters
    ----------
    cmd:
        Shell command string to execute remotely.
    capture:
        If True (default), capture stdout/stderr and return them.
        If False, inherit the parent process's stdout/stderr (useful for
        streaming long-running output).

    Returns
    -------
    subprocess.CompletedProcess with returncode, stdout, stderr.
    """
    return subprocess.run(
        ["ssh", _target(), cmd],
        capture_output=capture,
        text=True,
    )


def rsync_push() -> subprocess.CompletedProcess[str]:
    """Sync local code to the cluster using rsync.

    Uses ``--delete`` so removed local files are also removed remotely.
    Excludes large/generated directories listed in ``RSYNC_EXCLUDES``.
    """
    exclude_flags: list[str] = []
    for pattern in RSYNC_EXCLUDES:
        exclude_flags += ["--exclude", pattern]

    src = str(PROJECT_ROOT).rstrip("/\\") + "/"
    dst = f"{_target()}:{REMOTE_REPO}/"

    return subprocess.run(
        ["rsync", "-az", "--delete", *exclude_flags, src, dst],
        capture_output=True,
        text=True,
    )


def rsync_pull_results(
    remote_dir: str = "results/",
    local_dir: str = "results/",
) -> subprocess.CompletedProcess[str]:
    """Pull summary CSVs and metadata from the cluster.

    Only downloads summary files, metadata, and lifecycle logs — not the
    per-chunk result CSVs (which can be hundreds of files).
    """
    src = f"{_target()}:{REMOTE_REPO}/{remote_dir.rstrip('/')}/"
    dst_path = PROJECT_ROOT / local_dir.rstrip("/\\")
    dst_path.mkdir(parents=True, exist_ok=True)
    dst = str(dst_path).rstrip("/\\") + "/"

    return subprocess.run(
        [
            "rsync",
            "-az",
            "--include=*/",
            "--include=*_summary*.csv",
            "--include=metadata.json",
            "--include=config.txt",
            "--include=lifecycle.jsonl",
            "--exclude=*",
            src,
            dst,
        ],
        capture_output=True,
        text=True,
    )
