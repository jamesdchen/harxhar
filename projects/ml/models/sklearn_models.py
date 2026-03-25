"""Sklearn-based model wrappers: Ridge, XGBoost, LightGBM, RandomForest."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from core.models.base import RollingRegressionModel

# ── Model defaults ──────────────────────────────────────────────────────
# Each entry: (sklearn_class, default_use_scaling, default_refit_frequency,
#              extra defaults applied to kwargs before construction)

_MODEL_SPECS: dict[str, tuple[type, bool, int, dict[str, Any]]] = {
    "ridge": (Ridge, True, 1, {}),
    "xgboost": (XGBRegressor, False, 5, {"tree_method": "hist", "n_jobs": -1}),
    "lightgbm": (LGBMRegressor, False, 5, {"n_jobs": -1, "verbose": -1}),
    "random_forest": (RandomForestRegressor, False, 5, {"n_jobs": -1}),
}


# ── Class factory ───────────────────────────────────────────────────────


def _make_model_class(spec_key: str) -> type[RollingRegressionModel]:
    """Generate a RollingRegressionModel subclass from _MODEL_SPECS."""
    _, default_scaling, default_refit, _ = _MODEL_SPECS[spec_key]

    class _Model(RollingRegressionModel):
        def __init__(
            self,
            train_win_periods: int,
            n_features: int,
            use_scaling: bool = default_scaling,
            feature_transform: Any | None = None,
            refit_frequency: int = default_refit,
            **model_kwargs: Any,
        ) -> None:
            cls, _, _, defaults = _MODEL_SPECS[spec_key]
            merged = {**defaults, **model_kwargs}
            super().__init__(
                model=cls(**merged),
                train_win_periods=train_win_periods,
                n_features=n_features,
                use_scaling=use_scaling,
                refit_frequency=refit_frequency,
                feature_transform=feature_transform,
            )

    # Set readable class name for isinstance checks and repr
    name = spec_key.title().replace("_", "") + "Model"
    _Model.__name__ = _Model.__qualname__ = name
    return _Model


RidgeModel = _make_model_class("ridge")
XGBoostModel = _make_model_class("xgboost")
LightGBMModel = _make_model_class("lightgbm")
RandomForestModel = _make_model_class("random_forest")
