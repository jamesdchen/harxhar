"""SSH and rsync utilities for remote HPC operations.

Loads cluster/project config from claude-hpc (clusters.yaml + project.yaml)
with env-var overrides (HPC_HOST, HPC_USER, HPC_REPO) taking highest priority.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from hpc import load_clusters_config, load_project_config

__all__ = [  # noqa: F822 — REMOTE_* are lazy via __getattr__
    "REMOTE_HOST",
    "REMOTE_USER",
    "REMOTE_REPO",
    "RSYNC_EXCLUDES",
    "PROJECT_ROOT",
    "ssh_run",
    "rsync_push",
    "rsync_pull_results",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Hardcoded fallback excludes (used when project.yaml has no rsync_exclude).
RSYNC_EXCLUDES = [
    ".git/",
    "results/",
    "__pycache__/",
    "*.pyc",
    ".mypy_cache/",
    "all30min/",
    ".claude/",
]

# ---------------------------------------------------------------------------
# Lazy config loading (avoids import-time file I/O failures)
# ---------------------------------------------------------------------------

_cfg_cache: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load and cache merged config from project.yaml + clusters.yaml."""
    global _cfg_cache
    if _cfg_cache is not None:
        return _cfg_cache

    project = load_project_config(PROJECT_ROOT / "project.yaml")
    clusters = load_clusters_config()
    cluster_name = project.get("cluster", "hoffman2")
    cluster = clusters.get(cluster_name, {})

    _cfg_cache = {
        "host": os.environ.get("HPC_HOST") or cluster.get("host", "hoffman2.idre.ucla.edu"),
        "user": os.environ.get("HPC_USER") or cluster.get("user", "jamesdc1"),
        "repo": os.environ.get("HPC_REPO")
        or project.get("remote_path", "/u/home/j/jamesdc1/project-cucuringu/harxhar"),
        "rsync_exclude": project.get("rsync_exclude", RSYNC_EXCLUDES),
    }
    return _cfg_cache


# ---------------------------------------------------------------------------
# Module-level attribute access for REMOTE_HOST / REMOTE_USER / REMOTE_REPO
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> str:
    _map = {"REMOTE_HOST": "host", "REMOTE_USER": "user", "REMOTE_REPO": "repo"}
    if name in _map:
        return _load_config()[_map[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _target() -> str:
    cfg = _load_config()
    return f"{cfg['user']}@{cfg['host']}"


def ssh_run(cmd: str, *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run *cmd* on the cluster via SSH."""
    return subprocess.run(
        ["ssh", _target(), cmd],
        capture_output=capture,
        text=True,
    )


def rsync_push() -> subprocess.CompletedProcess[str]:
    """Sync local code to the cluster using rsync.

    Uses ``--delete`` so removed local files are also removed remotely.
    Excludes patterns from project.yaml rsync_exclude (falls back to
    RSYNC_EXCLUDES).
    """
    cfg = _load_config()
    exclude_flags: list[str] = []
    for pattern in cfg["rsync_exclude"]:
        exclude_flags += ["--exclude", pattern]

    src = str(PROJECT_ROOT).rstrip("/\\") + "/"
    dst = f"{_target()}:{cfg['repo']}/"

    return subprocess.run(
        ["rsync", "-az", "--delete", *exclude_flags, src, dst],
        capture_output=True,
        text=True,
    )


def rsync_pull_results(
    remote_dir: str = "results/",
    local_dir: str = "results/",
) -> subprocess.CompletedProcess[str]:
    """Pull summary CSVs and metadata from the cluster."""
    cfg = _load_config()
    src = f"{_target()}:{cfg['repo']}/{remote_dir.rstrip('/')}/"
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
