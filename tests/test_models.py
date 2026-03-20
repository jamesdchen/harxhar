"""Tests for src.models: factory wiring, predict/update cycles, and PCA integration."""

import numpy as np
import pytest

from src.features import PCATransform
from src.models import (
    LightGBMModel,
    NaiveBaseline,
    RandomForestModel,
    RidgeModel,
    SARIMAXModel,
    XGBoostModel,
    create_model,
)

# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


class TestModelFactory:
    def test_create_ridge(self):
        m = create_model("ridge", train_win_periods=100, n_features=5)
        assert isinstance(m, RidgeModel)
        assert m.use_scaling is True

    def test_create_xgboost(self):
        m = create_model("xgboost", train_win_periods=100, n_features=5)
        assert isinstance(m, XGBoostModel)
        assert m.use_scaling is False

    def test_create_naive(self):
        m = create_model("naive", train_win_periods=100, n_features=5, naive_lag_index=2)
        assert isinstance(m, NaiveBaseline)
        assert m.lag_index == 2

    def test_create_sarimax(self):
        m = create_model("sarimax", train_win_periods=100, n_features=5)
        assert isinstance(m, SARIMAXModel)

    def test_create_lightgbm(self):
        m = create_model("lightgbm", train_win_periods=100, n_features=5)
        assert isinstance(m, LightGBMModel)
        assert m.use_scaling is False

    def test_create_random_forest(self):
        m = create_model("random_forest", train_win_periods=100, n_features=5)
        assert isinstance(m, RandomForestModel)
        assert m.use_scaling is False

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("nonexistent", train_win_periods=100, n_features=5)

    def test_feature_transform_passed_through(self):
        ft = PCATransform(n_components=2)
        m = create_model("ridge", train_win_periods=100, n_features=5, feature_transform=ft)
        assert m.feature_transform is ft


# ---------------------------------------------------------------------------
# Predict/Update Cycles
# ---------------------------------------------------------------------------


class TestModelPredictUpdateCycles:
    """Smoke tests: initialize -> predict -> update -> predict for each model type."""

    @pytest.fixture
    def train_data(self):
        rng = np.random.RandomState(42)
        n_feat, win = 3, 50
        return rng.randn(win, n_feat), rng.randn(win), rng.randn(n_feat), n_feat, win

    def test_ridge_predict_update_cycle(self, train_data):
        X_init, y_init, x_t, n_feat, win = train_data
        m = create_model("ridge", train_win_periods=win, n_features=n_feat, alpha=1.0)
        m.initialize(X_init, y_init)
        pred = m.predict(x_t)
        assert np.isfinite(pred)
        m.update(x_t, 0.5)
        pred2 = m.predict(x_t)
        assert np.isfinite(pred2)

    def test_xgboost_predict_update_cycle(self, train_data):
        X_init, y_init, x_t, n_feat, win = train_data
        m = create_model("xgboost", train_win_periods=win, n_features=n_feat)
        m.initialize(X_init, y_init)
        pred = m.predict(x_t)
        assert np.isfinite(pred)
        m.update(x_t, 0.5)
        pred2 = m.predict(x_t)
        assert np.isfinite(pred2)

    def test_lightgbm_predict_update_cycle(self, train_data):
        X_init, y_init, x_t, n_feat, win = train_data
        m = create_model("lightgbm", train_win_periods=win, n_features=n_feat)
        m.initialize(X_init, y_init)
        pred = m.predict(x_t)
        assert np.isfinite(pred)
        m.update(x_t, 0.5)
        pred2 = m.predict(x_t)
        assert np.isfinite(pred2)

    def test_random_forest_predict_update_cycle(self, train_data):
        X_init, y_init, x_t, n_feat, win = train_data
        m = create_model("random_forest", train_win_periods=win, n_features=n_feat)
        m.initialize(X_init, y_init)
        pred = m.predict(x_t)
        assert np.isfinite(pred)
        m.update(x_t, 0.5)
        pred2 = m.predict(x_t)
        assert np.isfinite(pred2)

    def test_naive_returns_correct_lag(self):
        m = create_model("naive", train_win_periods=10, n_features=5, naive_lag_index=2)
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        m.initialize(np.zeros((10, 5)), np.zeros(10))
        assert m.predict(x) == 30.0


# ---------------------------------------------------------------------------
# PCA + Model Integration
# ---------------------------------------------------------------------------


class TestModelWithPCATransform:
    def test_ridge_with_pca_predict_update(self):
        rng = np.random.RandomState(42)
        n_feat, win = 5, 50
        ft = PCATransform(n_components=2)
        m = create_model("ridge", train_win_periods=win, n_features=n_feat, feature_transform=ft, alpha=1.0)
        X_init = rng.randn(win, n_feat)
        y_init = rng.randn(win)
        m.initialize(X_init, y_init)

        x_t = rng.randn(n_feat)
        pred = m.predict(x_t)
        assert np.isfinite(pred)
        m.update(x_t, 0.5)
        pred2 = m.predict(x_t)
        assert np.isfinite(pred2)

    def test_pca_reduces_coef_dimension(self):
        rng = np.random.RandomState(42)
        n_feat, win = 5, 50
        ft = PCATransform(n_components=2)
        m = create_model("ridge", train_win_periods=win, n_features=n_feat, feature_transform=ft, alpha=1.0)
        m.initialize(rng.randn(win, n_feat), rng.randn(win))
        coefs = m.get_coefs()
        assert coefs is not None
        assert len(coefs) == 2  # n_components, not n_features
