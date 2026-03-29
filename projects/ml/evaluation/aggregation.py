"""ML-specific config parsing + re-exports of core aggregation utilities."""

from __future__ import annotations

from pathlib import Path

# Re-export core aggregation functions so existing ML imports continue to work.
from core.evaluation.aggregation import (  # noqa: F401
    filter_by_time,
    load_all_chunks,
    load_experiment_metadata,
    process_single_experiment,
)


def parse_config(exp_dir: str | Path) -> tuple[int, str, str]:
    """Parses experiment metadata, delegating to :func:`core.evaluation.aggregation.load_experiment_metadata`.

    Returns ``(exp_id, experiment_name, model_type)`` for backward compatibility.
    """
    meta = load_experiment_metadata(exp_dir)
    return meta["exp_id"], meta["experiment_name"], meta["model"]
