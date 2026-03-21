"""
YAML-based experiment configuration.

Allows defining experiments as declarative config files for reproducibility:

    python scripts/submit.py from-config experiments/my_experiment.yaml

Example YAML:
    name: ridge_feature_comparison
    mode: subgroup_analysis
    models: [ridge, xgboost]
    features: [har, pca]
    subgroups: [baseline, moments, liquidity]
    train_window: 500
    horizon: 1
    total_chunks: 100
    backend: slurm
    notes: "Comparing ridge vs xgboost across feature subgroups"
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class ExperimentConfig:
    """Parsed experiment configuration."""

    name: str
    mode: str
    models: list[str] = dataclasses.field(default_factory=lambda: ["ridge"])
    features: list[str] = dataclasses.field(default_factory=lambda: ["har"])
    subgroups: list[str] = dataclasses.field(default_factory=lambda: ["all"])
    train_window: int = 500
    horizon: int = 1
    n_components: int = 5
    ae_alpha: float = 0.5
    ae_epochs: int = 50
    ae_hidden: int = 0
    ae_weights_path: str | None = None
    total_chunks: int = 100
    backend: str = "slurm"
    result_dir: str | None = None
    no_naive: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load a config file, supporting YAML (.yaml/.yml) or JSON (.json)."""
    text = path.read_text()
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            # Fall back to a simple key: value parser for basic YAML
            return _parse_simple_yaml(text)
        return yaml.safe_load(text)
    elif suffix == ".json":
        return json.loads(text)
    else:
        # Try JSON first, then simple YAML
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML-like parser for basic key: value and key: [list] configs.

    Handles the subset of YAML used by experiment configs without requiring
    the pyyaml dependency.
    """
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        # Strip inline comments
        if " #" in value:
            value = value[: value.index(" #")].strip()

        # Handle quoted strings
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            result[key] = value[1:-1]
        # Handle lists: [a, b, c]
        elif value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
            result[key] = items
        # Handle booleans
        elif value.lower() in ("true", "yes"):
            result[key] = True
        elif value.lower() in ("false", "no"):
            result[key] = False
        # Handle None
        elif value.lower() in ("null", "none", "~"):
            result[key] = None
        # Handle numbers
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a YAML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    raw = _load_yaml_or_json(path)

    if "name" not in raw:
        raw["name"] = path.stem

    if "mode" not in raw:
        raise ValueError(f"Experiment config {path} must specify a 'mode' field")

    # Map config keys to ExperimentConfig fields
    field_names = {f.name for f in dataclasses.fields(ExperimentConfig)}
    filtered = {k: v for k, v in raw.items() if k.replace("-", "_") in field_names}
    # Normalize hyphenated keys
    normalized = {k.replace("-", "_"): v for k, v in filtered.items()}

    return ExperimentConfig(**normalized)
