"""ML-specific feature group definitions.

Central registry of ~50 exogenous features organized into subgroups:
moments, liquidity, market_ew, market_vw, sentiment, implied_vol, vol_demand.

Exports: ALL_FEATURES (complete set), SUBGROUPS (name → column list),
ALL_MODELS, FEATURE_TYPES, FULL_FEATURE_STRING (pipe-separated for CLI).
Source of truth: ``feature_groups.py``.
"""

from projects.ml.features.feature_groups import (
    ALL_FEATURES,
    ALL_MODELS,
    FEATURE_TYPES,
    FULL_FEATURE_STRING,
    SUBGROUPS,
)

__all__ = [
    "ALL_FEATURES",
    "ALL_MODELS",
    "FEATURE_TYPES",
    "FULL_FEATURE_STRING",
    "SUBGROUPS",
]
