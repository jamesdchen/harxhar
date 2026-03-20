"""Model registry and factory function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import config as cfg
from src.models.base import BaseModel, NaiveBaseline
from src.models.sarimax import SARIMAXModel
from src.models.sklearn_models import LightGBMModel, RandomForestModel, RidgeModel, XGBoostModel

if TYPE_CHECKING:
    from src.features.transforms import BaseFeatureTransform

MODEL_REGISTRY = {
    "ridge": {
        "class": RidgeModel,
        "defaults": {"use_scaling": True, "alpha": 1.0},
    },
    "xgboost": {
        "class": XGBoostModel,
        "defaults": {
            "use_scaling": False,
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "tree_method": "hist",
        },
    },
    "lightgbm": {
        "class": LightGBMModel,
        "defaults": {"use_scaling": False, "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    },
    "random_forest": {
        "class": RandomForestModel,
        "defaults": {"use_scaling": False, "n_estimators": 100, "max_depth": 3},
    },
    "sarimax": {
        "class": SARIMAXModel,
        "defaults": {
            "order": cfg.SARIMAX_ORDER,
            "seasonal_order": cfg.SARIMAX_SEASONAL_ORDER,
            "fit_window": cfg.SARIMAX_FIT_WINDOW,
            "refit_frequency": cfg.SARIMAX_REFIT_FREQUENCY,
        },
    },
}


def create_model(
    model_name: str,
    train_win_periods: int,
    n_features: int,
    feature_transform: BaseFeatureTransform | None = None,
    refit_frequency: int = 1,
    naive_lag_index: int | None = None,
    horizon: int = 1,
    **overrides,
) -> BaseModel:
    """
    Factory function that creates a model instance from the registry.

    Parameters
    ----------
    model_name : str
        Key in MODEL_REGISTRY, or 'naive' for NaiveBaseline.
    naive_lag_index : int or None
        Required when model_name == 'naive'.
    horizon : int
        Forecast horizon (used by SARIMAX for native multi-step).
    **overrides
        Override any default hyperparameter from the registry.
    """
    if model_name == "naive":
        return NaiveBaseline(lag_index=naive_lag_index)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_name}")

    entry = MODEL_REGISTRY[model_name]
    kwargs = {**entry["defaults"], **overrides}

    # SARIMAX uses its own refit_frequency, doesn't take feature_transform,
    # and supports native multi-step via horizon parameter
    if model_name == "sarimax":
        return entry["class"](
            train_win_periods=train_win_periods,
            n_features=n_features,
            horizon=horizon,
            **kwargs,
        )

    return entry["class"](
        train_win_periods=train_win_periods,
        n_features=n_features,
        feature_transform=feature_transform,
        refit_frequency=refit_frequency,
        **kwargs,
    )
