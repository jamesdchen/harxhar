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


def _make_sklearn_model(
    spec_key: str,
    train_win_periods: int,
    n_features: int,
    use_scaling: bool | None = None,
    refit_frequency: int | None = None,
    feature_transform: Any | None = None,
    **model_kwargs: Any,
) -> RollingRegressionModel:
    """Construct a RollingRegressionModel from a spec key and overrides."""
    cls, default_scaling, default_refit, defaults = _MODEL_SPECS[spec_key]
    merged = {**defaults, **model_kwargs}
    return RollingRegressionModel(
        model=cls(**merged),
        train_win_periods=train_win_periods,
        n_features=n_features,
        use_scaling=default_scaling if use_scaling is None else use_scaling,
        refit_frequency=default_refit if refit_frequency is None else refit_frequency,
        feature_transform=feature_transform,
    )


# ── Public thin wrappers (preserve existing API) ───────────────────────


def _init_from_spec(
    self, spec_key: str, train_win_periods, n_features, use_scaling, refit_frequency, feature_transform, **model_kwargs
):
    """Shared init logic: resolve defaults from _MODEL_SPECS and call super().__init__."""
    cls, default_scaling, default_refit, defaults = _MODEL_SPECS[spec_key]
    merged = {**defaults, **model_kwargs}
    super(type(self), self).__init__(
        model=cls(**merged),
        train_win_periods=train_win_periods,
        n_features=n_features,
        use_scaling=default_scaling if use_scaling is None else use_scaling,
        refit_frequency=default_refit if refit_frequency is None else refit_frequency,
        feature_transform=feature_transform,
    )


class RidgeModel(RollingRegressionModel):
    def __init__(
        self,
        train_win_periods: int,
        n_features: int,
        use_scaling: bool = True,
        feature_transform: Any | None = None,
        refit_frequency: int = 1,
        **ridge_kwargs: Any,
    ) -> None:
        _init_from_spec(
            self,
            "ridge",
            train_win_periods,
            n_features,
            use_scaling,
            refit_frequency,
            feature_transform,
            **ridge_kwargs,
        )


class XGBoostModel(RollingRegressionModel):
    def __init__(
        self,
        train_win_periods: int,
        n_features: int,
        use_scaling: bool = False,
        refit_frequency: int = 5,
        feature_transform: Any | None = None,
        **xgb_kwargs: Any,
    ) -> None:
        _init_from_spec(
            self,
            "xgboost",
            train_win_periods,
            n_features,
            use_scaling,
            refit_frequency,
            feature_transform,
            **xgb_kwargs,
        )


class LightGBMModel(RollingRegressionModel):
    def __init__(
        self,
        train_win_periods: int,
        n_features: int,
        use_scaling: bool = False,
        refit_frequency: int = 5,
        feature_transform: Any | None = None,
        **lgbm_kwargs: Any,
    ) -> None:
        _init_from_spec(
            self,
            "lightgbm",
            train_win_periods,
            n_features,
            use_scaling,
            refit_frequency,
            feature_transform,
            **lgbm_kwargs,
        )


class RandomForestModel(RollingRegressionModel):
    def __init__(
        self,
        train_win_periods: int,
        n_features: int,
        use_scaling: bool = False,
        refit_frequency: int = 5,
        feature_transform: Any | None = None,
        **rf_kwargs: Any,
    ) -> None:
        _init_from_spec(
            self,
            "random_forest",
            train_win_periods,
            n_features,
            use_scaling,
            refit_frequency,
            feature_transform,
            **rf_kwargs,
        )
