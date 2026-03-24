"""End-to-end integration tests and multihorizon forecasting tests."""

import argparse

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# End-to-End Smoke Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_load_and_prep_returns_correct_shapes(self, synthetic_data):
        from core.data import load_and_prep_data_strided

        hparams = {
            "exog_cols": None,
            "is_tree": False,
            "use_transform_exog": True,
            "use_diurnal": True,
            "use_winsor": True,
            "allow_missing": False,
            "feature_type": "raw",
        }
        X, y, dates, baselines, features = load_and_prep_data_strided(hparams, synthetic_data, lag=5)
        assert X.ndim == 2
        assert len(y) == len(X)
        assert len(dates) == len(X)
        assert len(baselines) == len(X)
        assert len(features) == X.shape[1]
        assert X.shape[0] > 0

    def test_load_and_prep_tree_model(self, synthetic_data):
        from core.data import load_and_prep_data_strided

        hparams = {
            "exog_cols": None,
            "is_tree": True,
            "use_transform_exog": False,
            "use_diurnal": False,
            "use_winsor": False,
            "allow_missing": False,
            "feature_type": "raw",
        }
        X, y, dates, baselines, features = load_and_prep_data_strided(hparams, synthetic_data, lag=5)
        assert "DOW" in features
        assert "hour" in features
        assert X.shape[0] > 0

    def test_backtest_smoke(self, synthetic_data):
        """Full pipeline: load data -> create model -> run backtest."""
        from core.backtest import run_backtest_agnostic
        from core.data import load_and_prep_data_strided
        from projects.ml.models import create_model

        hparams = {
            "exog_cols": None,
            "is_tree": False,
            "use_transform_exog": True,
            "use_diurnal": True,
            "use_winsor": False,
            "allow_missing": False,
            "feature_type": "raw",
        }
        X, y, dates, baselines, features = load_and_prep_data_strided(hparams, synthetic_data, lag=5)

        train_win = 200
        assert X.shape[0] > train_win + 10, "Not enough data for backtest"

        model = create_model("ridge", train_win_periods=train_win, n_features=X.shape[1], alpha=1.0)

        test_indices = np.arange(train_win, min(train_win + 50, X.shape[0]))
        preds, coefs = run_backtest_agnostic(model, test_indices, X, y, train_win, save_coefs=True)

        assert len(preds) == len(test_indices)
        assert np.all(np.isfinite(preds))
        assert coefs is not None
        assert coefs.shape == (len(test_indices), X.shape[1])

    def test_backtest_naive(self, synthetic_data):
        from core.backtest import run_backtest_agnostic
        from core.data import load_and_prep_data_strided
        from projects.ml.models import create_model

        hparams = {
            "exog_cols": None,
            "is_tree": False,
            "allow_missing": False,
            "feature_type": "raw",
        }
        X, y, dates, baselines, features = load_and_prep_data_strided(hparams, synthetic_data, lag=5)

        train_win = 200
        model = create_model("naive", train_win_periods=train_win, n_features=X.shape[1], naive_lag_index=0)

        test_indices = np.arange(train_win, min(train_win + 20, X.shape[0]))
        preds, _ = run_backtest_agnostic(model, test_indices, X, y, train_win)

        # Naive should return X[t, 0] for each t
        for i, t_idx in enumerate(test_indices):
            assert preds[i] == X[t_idx, 0], f"Naive mismatch at step {i}"

    def test_save_and_load_results(self, synthetic_data, tmp_path):
        """Test that save_chunk_results produces valid CSV."""
        from core.backtest import save_chunk_results

        n = 100
        rng = np.random.RandomState(42)
        forecasts = rng.randn(n)
        indices = np.arange(200, 200 + n)
        y_true = rng.randn(300)
        dates = pd.Series(pd.date_range("2020-01-01", periods=300, freq="h"))
        baselines = np.ones(300)

        out = str(tmp_path / "results_chunk_1.csv")
        save_chunk_results(out, forecasts, indices, 200, y_true, dates, baselines)

        df = pd.read_csv(out)
        assert len(df) == n
        assert set(df.columns) == {"date", "horizon", "true_adj", "pred_adj", "true_raw", "pred_raw"}

    def test_get_chunk_indices(self):
        from core.backtest import get_chunk_indices_strided

        X = np.zeros((1000, 5))
        indices = get_chunk_indices_strided(X, train_window_size=200, chunk_id=0, total_chunks=4)
        assert len(indices) > 0
        assert indices[0] == 200

        # All chunks together should cover all test indices
        all_idx = []
        for i in range(4):
            all_idx.extend(get_chunk_indices_strided(X, 200, i, 4).tolist())
        assert sorted(all_idx) == list(range(200, 1000))

    def test_hparams_wiring(self):
        """Verify get_common_hparams sets all keys consumed by load_and_clean_base_data."""
        from projects.ml.cli.executor import get_common_hparams

        required_keys = [
            "is_tree",
            "use_transform_exog",
            "use_diurnal",
            "use_winsor",
            "allow_missing",
            "exog_cols",
            "feature_type",
        ]

        for model in ["ridge", "xgboost", "lightgbm", "random_forest", "sarimax"]:
            args = argparse.Namespace(model=model, features="har", exog_cols=None, lag_scope="global")
            hp = get_common_hparams(args)
            for key in required_keys:
                assert key in hp, f"Missing key '{key}' for model '{model}'"


# ---------------------------------------------------------------------------
# Multihorizon Forecasting
# ---------------------------------------------------------------------------


class TestHorizonShift:
    def test_horizon_1_is_identity(self):
        """horizon=1 should return data unchanged."""
        from core.data import apply_horizon_shift

        X = np.arange(20).reshape(10, 2).astype(float)
        y = np.arange(10, dtype=float)
        dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="h"))
        baselines = np.ones(10)

        X_h, y_h, dates_h, baselines_h = apply_horizon_shift(X, y, dates, baselines, 1)
        np.testing.assert_array_equal(X_h, X)
        np.testing.assert_array_equal(y_h, y)
        assert len(dates_h) == 10

    def test_horizon_shift_alignment(self):
        """With horizon=h, y[t] should equal original y[t + h - 1]."""
        from core.data import apply_horizon_shift

        N = 20
        X = np.arange(N * 3).reshape(N, 3).astype(float)
        y = np.arange(N, dtype=float) * 10
        dates = pd.Series(pd.date_range("2020-01-01", periods=N, freq="h"))
        baselines = np.arange(N, dtype=float)

        for h in [2, 4, 8]:
            X_h, y_h, dates_h, baselines_h = apply_horizon_shift(X, y, dates, baselines, h)
            shift = h - 1
            expected_len = N - shift
            assert len(y_h) == expected_len, f"h={h}: expected {expected_len}, got {len(y_h)}"
            assert len(X_h) == expected_len
            # y_h[t] should equal original y[t + shift]
            for t in range(expected_len):
                assert y_h[t] == y[t + shift], f"h={h}, t={t}: {y_h[t]} != {y[t + shift]}"
            # X_h[t] should equal original X[t] (prediction-time features)
            np.testing.assert_array_equal(X_h, X[:expected_len])
            # baselines aligned with target time
            np.testing.assert_array_equal(baselines_h, baselines[shift:])

    def test_horizon_boundary_validation(self):
        """horizon < 1 or > 48 should raise ValueError."""
        from core.data import apply_horizon_shift

        X = np.zeros((10, 2))
        y = np.zeros(10)
        dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="h"))
        baselines = np.ones(10)

        with pytest.raises(ValueError):
            apply_horizon_shift(X, y, dates, baselines, 0)
        with pytest.raises(ValueError):
            apply_horizon_shift(X, y, dates, baselines, 49)

    def test_backtest_multihorizon_smoke(self, synthetic_data):
        """End-to-end: load data -> horizon shift -> backtest for h=1..4."""
        from core.backtest import run_backtest_agnostic
        from core.data import apply_horizon_shift, load_and_prep_data_strided
        from projects.ml.models import create_model

        hparams = {
            "exog_cols": None,
            "is_tree": False,
            "use_transform_exog": True,
            "use_diurnal": True,
            "use_winsor": False,
            "allow_missing": False,
            "feature_type": "raw",
        }
        X, y, dates, baselines, features = load_and_prep_data_strided(hparams, synthetic_data, lag=5)

        train_win = 200
        assert X.shape[0] > train_win + 50, "Not enough data"

        prev_len = None
        for h in range(1, 5):
            X_h, y_h, dates_h, baselines_h = apply_horizon_shift(X, y, dates, baselines, h)
            assert len(X_h) == len(y_h)

            # Length should decrease by 1 for each additional horizon step
            if prev_len is not None:
                assert len(X_h) == prev_len - 1
            prev_len = len(X_h)

            model = create_model("ridge", train_win_periods=train_win, n_features=X_h.shape[1], alpha=1.0)
            test_indices = np.arange(train_win, min(train_win + 20, X_h.shape[0]))
            preds, _ = run_backtest_agnostic(model, test_indices, X_h, y_h, train_win)

            assert len(preds) == len(test_indices)
            assert np.all(np.isfinite(preds)), f"Non-finite predictions at h={h}"

    def test_sarimax_horizon_parameter(self):
        """Verify SARIMAX model accepts horizon parameter in factory."""
        from projects.ml.models import SARIMAXModel, create_model

        m = create_model("sarimax", train_win_periods=100, n_features=5, horizon=4)
        assert isinstance(m, SARIMAXModel)
        assert m.model.horizon == 4

    def test_results_include_horizon_column(self, tmp_path):
        """Verify save_chunk_results includes horizon in output CSV."""
        from core.backtest import save_chunk_results

        n = 50
        rng = np.random.RandomState(42)
        forecasts = rng.randn(n)
        indices = np.arange(100, 100 + n)
        y_true = rng.randn(200)
        dates = pd.Series(pd.date_range("2020-01-01", periods=200, freq="h"))
        baselines = np.ones(200)

        out = str(tmp_path / "results_chunk_1.csv")
        save_chunk_results(out, forecasts, indices, 100, y_true, dates, baselines, horizon=4)

        df = pd.read_csv(out)
        assert "horizon" in df.columns
        assert (df["horizon"] == 4).all()
