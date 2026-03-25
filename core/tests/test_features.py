"""Tests for src.features: HAR/Raw lag generation and PCA transform."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from core.features import PCATransform

# ---------------------------------------------------------------------------
# Feature Generation
# ---------------------------------------------------------------------------


class TestHARFeatures:
    def test_rolling_mean_matches_pandas(self):
        from core.features import HARFeatures

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
        from core.features import HARFeatures

        gen = HARFeatures(lags=[1, 5], target_col="adj_RV")
        series = pd.Series(np.arange(20, dtype=float))
        df = pd.DataFrame({"adj_RV": series})

        feat_dict, _ = gen.generate_pandas(df, ["adj_RV"])
        expected = np.column_stack([feat_dict["har_ma_1"], feat_dict["har_ma_5"]])

        result = gen.transform(series.values.reshape(-1, 1))
        # Both should have NaN at position 0 (shift), compare non-NaN
        mask = ~np.isnan(expected[:, 0])
        np.testing.assert_allclose(result[mask], expected[mask], atol=1e-10)

    def test_multi_column_feature_names(self):
        from core.features import HARFeatures

        gen = HARFeatures(lags=[1, 5], target_col="adj_RV")
        df = pd.DataFrame({"adj_RV": np.arange(10.0), "adj_exog": np.arange(10.0) * 2})
        _, names = gen.generate_pandas(df, ["adj_RV", "adj_exog"])
        assert names == ["har_ma_1", "har_ma_5", "adj_exog_ma_1", "adj_exog_ma_5"]

    def test_first_value_is_nan(self):
        from core.features import HARFeatures

        gen = HARFeatures(lags=[1], target_col="adj_RV")
        df = pd.DataFrame({"adj_RV": [10.0, 20.0, 30.0]})
        feat_dict, _ = gen.generate_pandas(df, ["adj_RV"])
        assert np.isnan(feat_dict["har_ma_1"].iloc[0])


class TestRawLagFeatures:
    def test_shift_matches_pandas(self):
        from core.features import RawLagFeatures

        gen = RawLagFeatures(lags=[1, 3], target_col="adj_RV")
        series = pd.Series(np.arange(10, dtype=float))
        df = pd.DataFrame({"adj_RV": series})

        feat_dict, names = gen.generate_pandas(df, ["adj_RV"])
        assert "adj_RV_lag_1" in names
        assert "adj_RV_lag_3" in names

        pd.testing.assert_series_equal(pd.Series(feat_dict["adj_RV_lag_1"]), series.shift(1), check_names=False)
        pd.testing.assert_series_equal(pd.Series(feat_dict["adj_RV_lag_3"]), series.shift(3), check_names=False)

    def test_numpy_transform_matches_pandas(self):
        from core.features import RawLagFeatures

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
# PCA Transform (from test_features_extended.py)
# ---------------------------------------------------------------------------


class TestPCATransform:
    def test_fit_reduces_dimensions(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        pca = PCATransform(n_components=3)
        pca.fit(X)
        result = pca.transform(X)
        assert result.shape == (100, 3)

    def test_explained_variance_decreasing(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        pca = PCATransform(n_components=5)
        pca.fit(X)
        ev = pca.pca.explained_variance_
        assert np.all(ev[:-1] >= ev[1:])

    def test_transform_without_fit_raises(self):
        pca = PCATransform(n_components=3)
        X = np.random.randn(10, 5)
        with pytest.raises((NotFittedError, AttributeError)):
            pca.transform(X)

    def test_single_component(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        pca = PCATransform(n_components=1)
        pca.fit(X)
        result = pca.transform(X)
        assert result.shape == (50, 1)

    def test_fit_transform_preserves_samples(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 8)
        pca = PCATransform(n_components=3)
        pca.fit(X)
        out = pca.transform(X[:5])
        assert out.shape == (5, 3)

    def test_roundtrip_low_rank(self):
        """PCA(n_components=2) on rank-2 data should capture nearly all variance."""
        rng = np.random.RandomState(42)
        basis = rng.randn(2, 5)
        coeffs = rng.randn(100, 2)
        X = coeffs @ basis  # rank-2 in 5D
        pca = PCATransform(n_components=2)
        pca.fit(X)
        assert sum(pca.pca.explained_variance_ratio_) > 0.99
