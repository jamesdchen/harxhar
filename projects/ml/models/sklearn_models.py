"""Sklearn-based model wrappers: Ridge, XGBoost, LightGBM, RandomForest."""

from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from core.models.base import RollingRegressionModel


class RidgeModel(RollingRegressionModel):
    def __init__(
        self, train_win_periods, n_features, use_scaling=True, feature_transform=None, refit_frequency=1, **ridge_kwargs
    ):
        model = Ridge(**ridge_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
        )


class XGBoostModel(RollingRegressionModel):
    def __init__(
        self, train_win_periods, n_features, use_scaling=False, refit_frequency=5, feature_transform=None, **xgb_kwargs
    ):
        if "tree_method" not in xgb_kwargs:
            xgb_kwargs["tree_method"] = "hist"
        if "n_jobs" not in xgb_kwargs:
            xgb_kwargs["n_jobs"] = -1

        model = XGBRegressor(**xgb_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
        )


class LightGBMModel(RollingRegressionModel):
    def __init__(
        self, train_win_periods, n_features, use_scaling=False, refit_frequency=5, feature_transform=None, **lgbm_kwargs
    ):
        if "n_jobs" not in lgbm_kwargs:
            lgbm_kwargs["n_jobs"] = -1
        if "verbose" not in lgbm_kwargs:
            lgbm_kwargs["verbose"] = -1

        model = LGBMRegressor(**lgbm_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
        )


class RandomForestModel(RollingRegressionModel):
    def __init__(
        self, train_win_periods, n_features, use_scaling=False, refit_frequency=5, feature_transform=None, **rf_kwargs
    ):
        if "n_jobs" not in rf_kwargs:
            rf_kwargs["n_jobs"] = -1

        model = RandomForestRegressor(**rf_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
        )
