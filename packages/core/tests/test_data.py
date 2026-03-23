"""Tests for src.data: rolling infrastructure, transforms, and edge cases."""

import numpy as np
import pandas as pd

from harxhar_core.data import apply_data_transform
from harxhar_core.features import resolve_lags

# ---------------------------------------------------------------------------
# Rolling Infrastructure
# ---------------------------------------------------------------------------


class TestRollingBuffer:
    def test_add_wraps_correctly(self):
        from harxhar_core.data.rolling import RollingBuffer

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
        from harxhar_core.data.rolling import RollingBuffer

        buf = RollingBuffer(window_size=3, n_features=1, n_targets=1)
        for i in range(5):
            buf.add(np.array([float(i)]), float(i * 10))
        X, y = buf.get_ordered_view()
        # Should be chronological: 2, 3, 4
        np.testing.assert_array_equal(X.ravel(), [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(y.ravel(), [20.0, 30.0, 40.0])

    def test_get_ordered_view_no_wrap(self):
        """When ptr=0 (just filled or never wrapped), ordered = raw."""
        from harxhar_core.data.rolling import RollingBuffer

        buf = RollingBuffer(window_size=3, n_features=1, n_targets=1)
        for i in range(3):
            buf.add(np.array([float(i)]), float(i))
        # ptr wraps to 0 after exactly window_size adds
        X_ord, _ = buf.get_ordered_view()
        X_raw, _ = buf.get_view()
        np.testing.assert_array_equal(X_ord, X_raw)


class TestRollingRobustScaler:
    def test_median_iqr_known_values(self):
        from harxhar_core.data.rolling import RollingRobustScaler

        scaler = RollingRobustScaler(window_size=5, n_features=1)
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        scaler.initialize(data)
        median, iqr = scaler.get_scaler()
        assert abs(median[0] - 3.0) < 0.01
        assert iqr[0] > 0

    def test_update_maintains_sorted_invariant(self):
        from harxhar_core.data.rolling import RollingRobustScaler

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
        from harxhar_core.data.rolling import RollingRobustScaler

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
# Data Transforms
# ---------------------------------------------------------------------------


class TestDataTransforms:
    def test_apply_data_transform_sqrt(self):
        s = pd.Series([1.0, 4.0, 9.0])
        result = apply_data_transform(s, "RV", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0])

    def test_apply_data_transform_log(self):
        s = pd.Series([1.0, np.e, np.e**2])
        result = apply_data_transform(s, "some_col", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [0.0, 1.0, 2.0])

    def test_apply_data_transform_signed_sqrt(self):
        s = pd.Series([4.0, -4.0, 0.0])
        result = apply_data_transform(s, "autocov_x", has_negatives=True, allow_missing=False)
        np.testing.assert_allclose(result.values, [2.0, -2.0, 0.0])

    def test_rolling_winsorize_clips(self):
        from harxhar_core.data import rolling_winsorize

        s = pd.Series([1.0] * 20 + [100.0])
        result = rolling_winsorize(s, window=20, allow_missing=False, is_target=False)
        # The outlier at index 20 should be clipped
        assert result.iloc[-1] < 100.0

    def test_diurnal_adjust_baseline_nonzero(self):
        from harxhar_core.data import diurnal_adjust

        n = 100
        series = pd.Series(np.random.RandomState(42).rand(n) + 1.0)
        tod = pd.Series(np.arange(n) % 10)  # 10 time slots
        adjusted, baseline = diurnal_adjust(series, tod, has_negatives=False, window=20, min_periods=5)
        assert len(adjusted) == n
        assert len(baseline) == n
        assert not baseline.isna().all()

    def test_robust_transform_skip_vars(self):
        from harxhar_core.data import robust_transform

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
# Edge Cases (from test_data_edge_cases.py)
# ---------------------------------------------------------------------------


class TestApplyDataTransformEdgeCases:
    def test_ret3_cube_root(self):
        s = pd.Series([8.0, 27.0])
        result = apply_data_transform(s, "ret3_col", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [2.0, 3.0])

    def test_ret4_fourth_root(self):
        s = pd.Series([16.0, 81.0])
        result = apply_data_transform(s, "ret4_col", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [2.0, 3.0])

    def test_has_negatives_fillna(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = apply_data_transform(s, "signed_col", has_negatives=True, allow_missing=False)
        assert result.iloc[1] == 0.0

    def test_has_negatives_allow_missing(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = apply_data_transform(s, "signed_col", has_negatives=True, allow_missing=True)
        assert np.isnan(result.iloc[1])


class TestResolveLags:
    def test_har_geometric_sequence(self):
        lags = resolve_lags("har", 125)
        assert lags == [1, 5, 25, 125]

    def test_har_partial(self):
        lags = resolve_lags("har", 10)
        assert lags == [1, 5]

    def test_raw_consecutive(self):
        lags = resolve_lags("raw", 5)
        assert lags == [1, 2, 3, 4, 5]
