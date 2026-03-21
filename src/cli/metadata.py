"""
Experiment metadata tracking.

Records git hash, timestamp, Python version, and full experiment config
alongside results for reproducibility.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _get_git_info() -> dict[str, str | bool]:
    """Capture current git state."""
    info: dict[str, str | bool] = {}
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["git_hash"] = git_hash
        info["git_short_hash"] = git_hash[:8]
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        # Check for uncommitted changes
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_dirty"] = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_hash"] = "unknown"
        info["git_short_hash"] = "unknown"
        info["git_branch"] = "unknown"
        info["git_dirty"] = False
    return info


def build_metadata(experiment_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a metadata dict capturing the current environment and config."""
    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    meta.update(_get_git_info())
    if experiment_config:
        meta["experiment_config"] = experiment_config
    return meta


def save_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Write metadata to a JSON file in the experiment directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return meta_path


def load_metadata(experiment_dir: str | Path) -> dict[str, Any] | None:
    """Load metadata from an experiment directory, if it exists."""
    meta_path = Path(experiment_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None
