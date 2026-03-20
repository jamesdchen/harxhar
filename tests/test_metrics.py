"""Tests for src/metrics.py — calculate_global_metrics and calculate_baseline_deltas."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_baseline_deltas, calculate_global_metrics, winsorize_series


class TestCalculateGlobalMetrics:
    def test_mse_mae_known_values(self):
        df = pd.DataFrame(
            {
                "true_adj": [1.0, 2.0, 3.0, 4.0, 5.0],
                "pred_adj": [2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        m = calculate_global_metrics(df)
        assert m["mse"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(1.0)

    def test_qlike_perfect_forecast(self):
        df = pd.DataFrame(
            {
                "true_raw": [1.0, 2.0, 4.0],
                "pred_raw": [1.0, 2.0, 4.0],
            }
        )
        m = calculate_global_metrics(df)
        # QLIKE = mean(true/pred - log(true/pred) - 1) = mean(1 - 0 - 1) = 0
        assert m["qlike"] == pytest.approx(0.0)

    def test_qlike_filters_nonpositive(self):
        df = pd.DataFrame(
            {
                "true_raw": [1.0, -1.0, 2.0],
                "pred_raw": [1.0, 1.0, 2.0],
            }
        )
        m = calculate_global_metrics(df)
        # Row with true_raw=-1.0 should be excluded; remaining are perfect → QLIKE=0
        assert m["qlike"] == pytest.approx(0.0)

    def test_qlike_all_nonpositive_returns_nan(self):
        df = pd.DataFrame(
            {
                "true_raw": [-1.0, -2.0],
                "pred_raw": [-1.0, -2.0],
            }
        )
        m = calculate_global_metrics(df)
        assert np.isnan(m["qlike"])

    def test_missing_raw_columns_no_qlike(self):
        df = pd.DataFrame(
            {
                "true_adj": [1.0, 2.0],
                "pred_adj": [1.0, 2.0],
            }
        )
        m = calculate_global_metrics(df)
        assert "mse" in m
        assert "qlike" not in m

    def test_n_samples_correct(self):
        df = pd.DataFrame({"true_adj": [1.0] * 7, "pred_adj": [1.0] * 7})
        m = calculate_global_metrics(df)
        assert m["n_samples"] == 7

    def test_winsorized_metrics_present(self):
        df = pd.DataFrame(
            {
                "true_adj": [1.0, 2.0, 3.0, 4.0, 5.0],
                "pred_adj": [2.0, 3.0, 4.0, 5.0, 6.0],
                "true_raw": [1.0, 2.0, 3.0, 4.0, 5.0],
                "pred_raw": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        m = calculate_global_metrics(df)
        assert "w_mse" in m
        assert "w_mae" in m
        assert "w_qlike" in m

    def test_winsorized_clips_outliers(self):
        """With an extreme outlier, winsorized MSE should be less than raw MSE."""
        true = [1.0] * 99 + [1.0]
        pred = [1.0] * 99 + [100.0]  # one huge outlier
        df = pd.DataFrame({"true_adj": true, "pred_adj": pred})
        m = calculate_global_metrics(df)
        assert m["w_mse"] < m["mse"]
        assert m["w_mae"] < m["mae"]

    def test_winsorized_no_effect_uniform_errors(self):
        """When all errors are identical, winsorization should not change the result."""
        df = pd.DataFrame(
            {
                "true_adj": [1.0, 2.0, 3.0],
                "pred_adj": [2.0, 3.0, 4.0],
            }
        )
        m = calculate_global_metrics(df)
        assert m["w_mse"] == pytest.approx(m["mse"])
        assert m["w_mae"] == pytest.approx(m["mae"])


class TestWinsorizeSeries:
    def test_clips_extremes(self):
        # 95 normal values + 5 extreme outliers — the 95th percentile is well below 1000
        data = np.array([1.0] * 95 + [1000.0] * 5)
        result = winsorize_series(data)
        assert result.max() < 1000.0

    def test_identity_when_uniform(self):
        data = np.array([3.0] * 100)
        result = winsorize_series(data)
        np.testing.assert_array_equal(result, data)

    def test_custom_quantiles(self):
        data = np.arange(100, dtype=float)
        result = winsorize_series(data, lower_q=0.10, upper_q=0.90)
        assert result.min() >= 9.0
        assert result.max() <= 90.0


class TestCalculateBaselineDeltas:
    def test_deltas_computed_correctly(self):
        summary = pd.DataFrame(
            {
                "exp_id": [0, 1],
                "experiment_name": ["baseline", "my_model"],
                "segment": ["full", "full"],
                "mse": [2.0, 1.0],
                "mae": [1.5, 1.0],
                "qlike": [0.5, 0.3],
            }
        )
        result = calculate_baseline_deltas(summary)
        # delta = model - baseline
        assert result.loc[1, "delta_mse"] == pytest.approx(-1.0)
        assert result.loc[1, "delta_mae"] == pytest.approx(-0.5)
        assert result.loc[1, "delta_qlike"] == pytest.approx(-0.2)
        # oos_r2 = 1 - model_mse / baseline_mse = 1 - 1/2 = 0.5
        assert result.loc[1, "oos_r2"] == pytest.approx(0.5)

    def test_deltas_multi_horizon(self):
        """Each horizon should be compared against its own naive baseline horizon."""
        summary = pd.DataFrame(
            {
                "exp_id": [0, 0, 1, 1],
                "experiment_name": ["naive_baseline", "naive_baseline", "my_model", "my_model"],
                "segment": ["full", "full", "full", "full"],
                "horizon": [1, 2, 1, 2],
                "mse": [2.0, 4.0, 1.0, 3.0],
                "mae": [1.5, 3.0, 1.0, 2.0],
                "qlike": [0.5, 1.0, 0.3, 0.7],
            }
        )
        result = calculate_baseline_deltas(summary)
        model_rows = result[result["exp_id"] == 1].set_index("horizon")

        # horizon=1: delta_mse = 1.0 - 2.0 = -1.0
        assert model_rows.loc[1, "delta_mse"] == pytest.approx(-1.0)
        assert model_rows.loc[1, "oos_r2"] == pytest.approx(0.5)

        # horizon=2: delta_mse = 3.0 - 4.0 = -1.0 (NOT 3.0 - 2.0 = 1.0)
        assert model_rows.loc[2, "delta_mse"] == pytest.approx(-1.0)
        assert model_rows.loc[2, "oos_r2"] == pytest.approx(0.25)

    def test_winsorized_deltas_computed(self):
        summary = pd.DataFrame(
            {
                "exp_id": [0, 1],
                "experiment_name": ["baseline", "my_model"],
                "segment": ["full", "full"],
                "mse": [2.0, 1.0],
                "mae": [1.5, 1.0],
                "qlike": [0.5, 0.3],
                "w_mse": [1.8, 0.9],
                "w_mae": [1.4, 0.9],
                "w_qlike": [0.4, 0.25],
            }
        )
        result = calculate_baseline_deltas(summary)
        assert result.loc[1, "delta_w_mse"] == pytest.approx(-0.9)
        assert result.loc[1, "delta_w_mae"] == pytest.approx(-0.5)
        assert result.loc[1, "delta_w_qlike"] == pytest.approx(-0.15)
        assert result.loc[1, "w_oos_r2"] == pytest.approx(0.5)

    def test_no_baseline_returns_nan(self):
        summary = pd.DataFrame(
            {
                "exp_id": [5],
                "experiment_name": ["my_model"],
                "segment": ["full"],
                "mse": [1.0],
                "mae": [1.0],
                "qlike": [0.3],
            }
        )
        result = calculate_baseline_deltas(summary)
        assert np.isnan(result.loc[0, "delta_mse"])
        assert np.isnan(result.loc[0, "oos_r2"])
