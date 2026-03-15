"""Tests for src/metrics.py — calculate_global_metrics and calculate_baseline_deltas."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import calculate_global_metrics, calculate_baseline_deltas


class TestCalculateGlobalMetrics:
    def test_mse_mae_known_values(self):
        df = pd.DataFrame({
            'true_adj': [1.0, 2.0, 3.0, 4.0, 5.0],
            'pred_adj': [2.0, 3.0, 4.0, 5.0, 6.0],
        })
        m = calculate_global_metrics(df)
        assert m['mse'] == pytest.approx(1.0)
        assert m['mae'] == pytest.approx(1.0)

    def test_qlike_perfect_forecast(self):
        df = pd.DataFrame({
            'true_raw': [1.0, 2.0, 4.0],
            'pred_raw': [1.0, 2.0, 4.0],
        })
        m = calculate_global_metrics(df)
        # QLIKE = mean(true/pred - log(true/pred) - 1) = mean(1 - 0 - 1) = 0
        assert m['qlike'] == pytest.approx(0.0)

    def test_qlike_filters_nonpositive(self):
        df = pd.DataFrame({
            'true_raw': [1.0, -1.0, 2.0],
            'pred_raw': [1.0, 1.0, 2.0],
        })
        m = calculate_global_metrics(df)
        # Row with true_raw=-1.0 should be excluded; remaining are perfect → QLIKE=0
        assert m['qlike'] == pytest.approx(0.0)

    def test_qlike_all_nonpositive_returns_nan(self):
        df = pd.DataFrame({
            'true_raw': [-1.0, -2.0],
            'pred_raw': [-1.0, -2.0],
        })
        m = calculate_global_metrics(df)
        assert np.isnan(m['qlike'])

    def test_missing_raw_columns_no_qlike(self):
        df = pd.DataFrame({
            'true_adj': [1.0, 2.0],
            'pred_adj': [1.0, 2.0],
        })
        m = calculate_global_metrics(df)
        assert 'mse' in m
        assert 'qlike' not in m

    def test_n_samples_correct(self):
        df = pd.DataFrame({'true_adj': [1.0] * 7, 'pred_adj': [1.0] * 7})
        m = calculate_global_metrics(df)
        assert m['n_samples'] == 7


class TestCalculateBaselineDeltas:
    def test_deltas_computed_correctly(self):
        summary = pd.DataFrame({
            'exp_id': [0, 1],
            'experiment_name': ['baseline', 'my_model'],
            'segment': ['full', 'full'],
            'mse': [2.0, 1.0],
            'mae': [1.5, 1.0],
            'qlike': [0.5, 0.3],
        })
        result = calculate_baseline_deltas(summary)
        # delta = model - baseline
        assert result.loc[1, 'delta_mse'] == pytest.approx(-1.0)
        assert result.loc[1, 'delta_mae'] == pytest.approx(-0.5)
        assert result.loc[1, 'delta_qlike'] == pytest.approx(-0.2)
        # oos_r2 = 1 - model_mse / baseline_mse = 1 - 1/2 = 0.5
        assert result.loc[1, 'oos_r2'] == pytest.approx(0.5)

    def test_no_baseline_returns_nan(self):
        summary = pd.DataFrame({
            'exp_id': [5],
            'experiment_name': ['my_model'],
            'segment': ['full'],
            'mse': [1.0],
            'mae': [1.0],
            'qlike': [0.3],
        })
        result = calculate_baseline_deltas(summary)
        assert np.isnan(result.loc[0, 'delta_mse'])
        assert np.isnan(result.loc[0, 'oos_r2'])
