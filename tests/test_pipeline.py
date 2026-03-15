"""
Tests for the HARXHAR forecasting pipeline.

Covers:
  1. Rolling infrastructure (RollingBuffer, RollingRobustScaler)
  2. Feature generation (HAR, Raw lags)
  3. Model factory wiring
  4. Data transform pipeline
  5. End-to-end smoke test
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. Rolling Infrastructure
# ---------------------------------------------------------------------------


class TestRollingBuffer:
    def test_add_wraps_correctly(self):
        from src.data.rolling import RollingBuffer

        buf = RollingBuffer(window_size=3, n_features=2, n_targets=1)
        # Fill buffer past capacity
        for i in range(5):
            buf.add(np.array([i, i * 10], dtype=np.float32), np.array([i * 100], dtype=np.float32))
        # Buffer should contain the last 3 entries (indices 2,3,4)
        X, y = buf.get_view()
        assert X.shape == (3, 2)
        assert y.shape == (3, 1)
        # ptr should be at 2 (5 % 3 = 2)
        assert buf.ptr == 2

    def test_get_ordered_view_chronological(self):
        from src.data.rolling import RollingBuffer

        buf = RollingBuffer(window_size=3, n_features=1, n_targets=1)
        for i in range(5):
            buf.add(np.array([float(i)]), float(i * 10))
        X, y = buf.get_ordered_view()
        # Should be chronological: 2, 3, 4
        np.testing.assert_array_equal(X.ravel(), [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(y.ravel(), [20.0, 30.0, 40.0])

    def test_get_ordered_view_no_wrap(self):
        """When ptr=0 (just filled or never wrapped), ordered = raw."""
        from src.data.rolling import RollingBuffer

        buf = RollingBuffer(window_size=3, n_features=1, n_targets=1)
        for i in range(3):
            buf.add(np.array([float(i)]), float(i))
        # ptr wraps to 0 after exactly window_size adds
        X_ord, _ = buf.get_ordered_view()
        X_raw, _ = buf.get_view()
        np.testing.assert_array_equal(X_ord, X_raw)


class TestRollingRobustScaler:
    def test_median_iqr_known_values(self):
        from src.data.rolling import RollingRobustScaler

        scaler = RollingRobustScaler(window_size=5, n_features=1)
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        scaler.initialize(data)
        median, iqr = scaler.get_scaler()
        assert abs(median[0] - 3.0) < 0.01
        assert iqr[0] > 0

    def test_update_maintains_sorted_invariant(self):
        from src.data.rolling import RollingRobustScaler

        scaler = RollingRobustScaler(window_size=4, n_features=1)
        data = np.array([[1.0], [3.0], [5.0], [7.0]])
        scaler.initialize(data)
        # Replace oldest (1.0) with 10.0
        scaler.update(np.array([10.0]))
        # Sorted buffer should be [3, 5, 7, 10]
        sorted_vals = scaler.sorted_buffer[0]
        assert np.all(sorted_vals[:-1] <= sorted_vals[1:]), f"Sorted invariant violated: {sorted_vals}"

    def test_scaler_matches_numpy(self):
        """Scaler stats should match numpy after several updates."""
        from src.data.rolling import RollingRobustScaler

        rng = np.random.RandomState(42)
        window = 20
        n_feat = 3
        scaler = RollingRobustScaler(window, n_feat)
        init_data = rng.randn(window, n_feat)
        scaler.initialize(init_data)

        # Run 50 updates
        all_data = list(init_data)
        for _ in range(50):
            x = rng.randn(n_feat)
            scaler.update(x)
            all_data.append(x)

        # Compare against numpy on last `window` rows
        recent = np.array(all_data[-window:])
        expected_median = np.median(recent, axis=0)
        expected_iqr = np.percentile(recent, 75, axis=0) - np.percentile(recent, 25, axis=0)
        expected_iqr = np.where(expected_iqr < 1e-12, 1.0, expected_iqr)

        actual_median, actual_iqr = scaler.get_scaler()
        np.testing.assert_allclose(actual_median, expected_median, atol=0.05)
        np.testing.assert_allclose(actual_iqr, expected_iqr, atol=0.05)


# ---------------------------------------------------------------------------
# 2. Feature Generation
# ---------------------------------------------------------------------------


class TestHARFeatures:
    def test_rolling_mean_matches_pandas(self):
        from src.features import HARFeatures

        gen = HARFeatures(lags=[1, 5], target_col="adj_RV")
        series = pd.Series(np.arange(20, dtype=float))
        df = pd.DataFrame({"adj_RV": series})

        feat_dict, names = gen.generate_pandas(df, ["adj_RV"])
        assert "har_ma_1" in names
        assert "har_ma_5" in names

        # har_ma_1 with lag=1: rolling(1).mean().shift(1) == shift(1)
        expected_ma1 = series.rolling(1).mean().shift(1)
        pd.testing.assert_series_equal(pd.Series(feat_dict["har_ma_1"]), expected_ma1, check_names=False)

        # har_ma_5 with lag=5: rolling(5).mean().shift(1)
        expected_ma5 = series.rolling(5, min_periods=1).mean().shift(1)
        pd.testing.assert_series_equal(pd.Series(feat_dict["har_ma_5"]), expected_ma5, check_names=False)

    def test_numpy_transform_matches_pandas(self):
        from src.features import HARFeatures

        gen = HARFeatures(lags=[1, 5], target_col="adj_RV")
        series = pd.Series(np.arange(20, dtype=float))
        df = pd.DataFrame({"adj_RV": series})

        feat_dict, _ = gen.generate_pandas(df, ["adj_RV"])
        expected = np.column_stack([feat_dict["har_ma_1"], feat_dict["har_ma_5"]])

        result = gen.transform(series.values.reshape(-1, 1))
        # Both should have NaN at position 0 (shift), compare non-NaN
        mask = ~np.isnan(expected[:, 0])
        np.testing.assert_allclose(result[mask], expected[mask], atol=1e-10)


class TestRawLagFeatures:
    def test_shift_matches_pandas(self):
        from src.features import RawLagFeatures

        gen = RawLagFeatures(lags=[1, 3], target_col="adj_RV")
        series = pd.Series(np.arange(10, dtype=float))
        df = pd.DataFrame({"adj_RV": series})

        feat_dict, names = gen.generate_pandas(df, ["adj_RV"])
        assert "adj_RV_lag_1" in names
        assert "adj_RV_lag_3" in names

        pd.testing.assert_series_equal(pd.Series(feat_dict["adj_RV_lag_1"]), series.shift(1), check_names=False)
        pd.testing.assert_series_equal(pd.Series(feat_dict["adj_RV_lag_3"]), series.shift(3), check_names=False)

    def test_numpy_transform_matches_pandas(self):
        from src.features import RawLagFeatures

        gen = RawLagFeatures(lags=[1, 2], target_col="x")
        arr = np.arange(10, dtype=float)
        result = gen.transform(arr.reshape(-1, 1))
        # lag_1: [nan, 0, 1, ..., 8], lag_2: [nan, nan, 0, 1, ..., 7]
        assert np.isnan(result[0, 0])
        assert result[1, 0] == 0.0
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])
        assert result[2, 1] == 0.0


# ---------------------------------------------------------------------------
# 3. Model Factory Wiring
# ---------------------------------------------------------------------------


class TestModelFactory:
    def test_create_ridge(self):
        from src.models import RidgeModel, create_model

        m = create_model("ridge", train_win_periods=100, n_features=5)
        assert isinstance(m, RidgeModel)
        assert m.use_scaling is True

    def test_create_xgboost(self):
        from src.models import XGBoostModel, create_model

        m = create_model("xgboost", train_win_periods=100, n_features=5)
        assert isinstance(m, XGBoostModel)
        assert m.use_scaling is False

    def test_create_naive(self):
        from src.models import NaiveBaseline, create_model

        m = create_model("naive", train_win_periods=100, n_features=5, naive_lag_index=2)
        assert isinstance(m, NaiveBaseline)
        assert m.lag_index == 2

    def test_create_sarimax(self):
        from src.models import SARIMAXModel, create_model

        m = create_model("sarimax", train_win_periods=100, n_features=5)
        assert isinstance(m, SARIMAXModel)

    def test_unknown_model_raises(self):
        from src.models import create_model

        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("nonexistent", train_win_periods=100, n_features=5)

    def test_feature_transform_passed_through(self):
        from src.features import PCATransform
        from src.models import create_model

        ft = PCATransform(n_components=2)
        m = create_model("ridge", train_win_periods=100, n_features=5, feature_transform=ft)
        assert m.feature_transform is ft

    def test_ridge_predict_update_cycle(self):
        """Smoke test: initialize → predict → update doesn't crash."""
        from src.models import create_model

        rng = np.random.RandomState(42)
        n_feat = 3
        win = 50
        m = create_model("ridge", train_win_periods=win, n_features=n_feat, alpha=1.0)

        X_init = rng.randn(win, n_feat)
        y_init = rng.randn(win)
        m.initialize(X_init, y_init)

        x_t = rng.randn(n_feat)
        pred = m.predict(x_t)
        assert np.isfinite(pred)

        m.update(x_t, 0.5)
        pred2 = m.predict(x_t)
        assert np.isfinite(pred2)

    def test_naive_returns_correct_lag(self):
        from src.models import create_model

        m = create_model("naive", train_win_periods=10, n_features=5, naive_lag_index=2)
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        m.initialize(np.zeros((10, 5)), np.zeros(10))
        assert m.predict(x) == 30.0


# ---------------------------------------------------------------------------
# 4. Data Transform Pipeline
# ---------------------------------------------------------------------------


class TestDataTransforms:
    def test_apply_data_transform_sqrt(self):
        from src.data import apply_data_transform

        s = pd.Series([1.0, 4.0, 9.0])
        result = apply_data_transform(s, "RV", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0])

    def test_apply_data_transform_log(self):
        from src.data import apply_data_transform

        s = pd.Series([1.0, np.e, np.e**2])
        result = apply_data_transform(s, "some_col", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [0.0, 1.0, 2.0])

    def test_apply_data_transform_signed_sqrt(self):
        from src.data import apply_data_transform

        s = pd.Series([4.0, -4.0, 0.0])
        result = apply_data_transform(s, "autocov_x", has_negatives=True, allow_missing=False)
        np.testing.assert_allclose(result.values, [2.0, -2.0, 0.0])

    def test_rolling_winsorize_clips(self):
        from src.data import rolling_winsorize

        s = pd.Series([1.0] * 20 + [100.0])
        result = rolling_winsorize(s, window=20, allow_missing=False, is_target=False)
        # The outlier at index 20 should be clipped
        assert result.iloc[-1] < 100.0

    def test_diurnal_adjust_baseline_nonzero(self):
        from src.data import diurnal_adjust

        n = 100
        series = pd.Series(np.random.RandomState(42).rand(n) + 1.0)
        tod = pd.Series(np.arange(n) % 10)  # 10 time slots
        adjusted, baseline = diurnal_adjust(series, tod, has_negatives=False, window=20, min_periods=5)
        assert len(adjusted) == n
        assert len(baseline) == n
        assert not baseline.isna().all()

    def test_robust_transform_skip_vars(self):
        from src.data import robust_transform

        df = pd.DataFrame(
            {
                "t": pd.date_range("2020-01-01", periods=10, freq="h"),
                "time_of_day": [0] * 10,
                "hour": list(range(10)),
            }
        )
        df.index = range(10)
        result, baseline = robust_transform(df, "hour")
        # 'hour' is in SKIP_VARS, should return raw values
        pd.testing.assert_series_equal(result, df["hour"])


# ---------------------------------------------------------------------------
# 5. End-to-End Smoke Test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create a minimal synthetic parquet for pipeline testing."""
        rng = np.random.RandomState(42)
        n = 5000
        dates = pd.date_range("2006-01-01", periods=n, freq="30min")
        # Filter out weekends like the pipeline does
        mask = ~((dates.dayofweek == 5) | (dates.dayofweek == 6))
        dates = dates[mask][:3000]
        n = len(dates)

        df = pd.DataFrame(
            {
                "endbartime": dates,
                "sumret2": np.abs(rng.randn(n)) * 0.001 + 0.0001,
                "hour": dates.hour,
                "DOW": dates.dayofweek,
            }
        )
        path = tmp_path / "test.parquet"
        df.to_parquet(path, engine="pyarrow")
        return str(path)

    def test_load_and_prep_returns_correct_shapes(self, synthetic_data):
        from src.data import load_and_prep_data_strided

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
        from src.data import load_and_prep_data_strided

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
        """Full pipeline: load data → create model → run backtest."""
        from src.backtest import run_backtest_agnostic
        from src.data import load_and_prep_data_strided
        from src.models import create_model

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
        from src.backtest import run_backtest_agnostic
        from src.data import load_and_prep_data_strided
        from src.models import create_model

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
        from src.backtest import save_chunk_results

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
        from src.backtest import get_chunk_indices_strided

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
        import argparse

        from src.cli.executor import get_common_hparams

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
            args = argparse.Namespace(model=model, features="raw", exog_cols=None, lag_scope="global")
            hp = get_common_hparams(args)
            for key in required_keys:
                assert key in hp, f"Missing key '{key}' for model '{model}'"


# ---------------------------------------------------------------------------
# 6. Multihorizon Forecasting
# ---------------------------------------------------------------------------


class TestHorizonShift:
    def test_horizon_1_is_identity(self):
        """horizon=1 should return data unchanged."""
        from src.data import apply_horizon_shift

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
        from src.data import apply_horizon_shift

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
        from src.data import apply_horizon_shift

        X = np.zeros((10, 2))
        y = np.zeros(10)
        dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="h"))
        baselines = np.ones(10)

        with pytest.raises(ValueError):
            apply_horizon_shift(X, y, dates, baselines, 0)
        with pytest.raises(ValueError):
            apply_horizon_shift(X, y, dates, baselines, 49)

    def test_backtest_multihorizon_smoke(self, synthetic_data):
        """End-to-end: load data → horizon shift → backtest for h=1..4."""
        from src.backtest import run_backtest_agnostic
        from src.data import apply_horizon_shift, load_and_prep_data_strided
        from src.models import create_model

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

    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create a minimal synthetic parquet for pipeline testing."""
        rng = np.random.RandomState(42)
        n = 5000
        dates = pd.date_range("2006-01-01", periods=n, freq="30min")
        mask = ~((dates.dayofweek == 5) | (dates.dayofweek == 6))
        dates = dates[mask][:3000]
        n = len(dates)
        df = pd.DataFrame(
            {
                "endbartime": dates,
                "sumret2": np.abs(rng.randn(n)) * 0.001 + 0.0001,
                "hour": dates.hour,
                "DOW": dates.dayofweek,
            }
        )
        path = tmp_path / "test.parquet"
        df.to_parquet(path, engine="pyarrow")
        return str(path)

    def test_sarimax_horizon_parameter(self):
        """Verify SARIMAX model accepts horizon parameter in factory."""
        from src.models import SARIMAXModel, create_model

        m = create_model("sarimax", train_win_periods=100, n_features=5, horizon=4)
        assert isinstance(m, SARIMAXModel)
        assert m.model.horizon == 4

    def test_results_include_horizon_column(self, tmp_path):
        """Verify save_chunk_results includes horizon in output CSV."""
        from src.backtest import save_chunk_results

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
