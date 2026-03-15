"""Tests for model types not covered in test_pipeline.py."""

import numpy as np
import pytest

from src.features import PCATransform
from src.models import LightGBMModel, RandomForestModel, create_model


class TestModelFactoryExtended:
    def test_create_lightgbm(self):
        m = create_model("lightgbm", train_win_periods=100, n_features=5)
        assert isinstance(m, LightGBMModel)
        assert m.use_scaling is False

    def test_create_random_forest(self):
        m = create_model("random_forest", train_win_periods=100, n_features=5)
        assert isinstance(m, RandomForestModel)
        assert m.use_scaling is False


class TestModelPredictUpdateCycles:
    """Smoke tests: initialize → predict → update → predict for each model type."""

    @pytest.fixture
    def train_data(self):
        rng = np.random.RandomState(42)
        n_feat, win = 3, 50
        return rng.randn(win, n_feat), rng.randn(win), rng.randn(n_feat), n_feat, win

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
