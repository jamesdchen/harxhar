"""Backward-compatible re-export — SLURM backend now lives in core.backends.slurm."""

from core.backends.slurm import SlurmBackend  # noqa: F401

# ML-specific default script path
DEFAULT_SUBMISSION_SCRIPT = str(
    __import__("core.backends", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
    / "projects"
    / "ml"
    / "infra"
    / "slurm"
    / "submit_carc.slurm"
)
