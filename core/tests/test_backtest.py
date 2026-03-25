"""Tests for core/backtest/engine.py functions."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import (
    apply_duan_smearing,
    build_results_dataframe,
    extract_subset,
    get_chunk_indices_strided,
)


class TestGetChunkIndicesStrided:
    def test_chunk_id_ge_total_chunks_returns_empty(self):
        X = np.zeros((100, 5))
        result = get_chunk_indices_strided(X, train_window_size=10, chunk_id=4, total_chunks=4)
        assert len(result) == 0

    def test_train_window_ge_num_samples_returns_empty(self):
        X = np.zeros((10, 5))
        result = get_chunk_indices_strided(X, train_window_size=10, chunk_id=0, total_chunks=1)
        assert len(result) == 0

    def test_normal_first_index_equals_train_window(self):
        X = np.zeros((100, 5))
        result = get_chunk_indices_strided(X, train_window_size=20, chunk_id=0, total_chunks=2)
        assert result[0] == 20

    def test_chunks_cover_all_test_indices(self):
        X = np.zeros((50, 3))
        tw = 10
        total = 4
        all_idx = np.concatenate([get_chunk_indices_strided(X, tw, c, total) for c in range(total)])
        np.testing.assert_array_equal(all_idx, np.arange(tw, 50))


class TestApplyDuanSmearing:
    def test_known_values(self):
        forecasts = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.5, 2.5, 3.5])
        baselines = np.array([1.0, 1.0, 1.0])
        smear = np.mean((y_true - forecasts) ** 2)
        expected_pred = forecasts**2 + smear
        pred_raw, true_raw = apply_duan_smearing(forecasts, y_true, baselines)
        np.testing.assert_allclose(pred_raw, expected_pred)
        np.testing.assert_allclose(true_raw, y_true**2)

    def test_baselines_scale_output(self):
        forecasts = np.array([2.0, 3.0])
        y_true = np.array([2.0, 3.0])
        baselines = np.array([10.0, 20.0])
        pred_raw, true_raw = apply_duan_smearing(forecasts, y_true, baselines)
        # smear = 0 when forecasts == y_true
        np.testing.assert_allclose(pred_raw, forecasts**2 * baselines)
        np.testing.assert_allclose(true_raw, y_true**2 * baselines)

    def test_nan_input_raises(self):
        bad = np.array([1.0, np.nan, 3.0])
        ok = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            apply_duan_smearing(bad, ok, ok)
        with pytest.raises(ValueError):
            apply_duan_smearing(ok, bad, ok)
        with pytest.raises(ValueError):
            apply_duan_smearing(ok, ok, bad)


class TestBuildResultsDataframe:
    def test_correct_columns(self):
        n = 5
        fc = np.ones(n)
        y = np.ones(n) * 2
        dates = np.arange(n)
        bases = np.ones(n)
        df = build_results_dataframe(fc, y, dates, bases, horizon=1)
        assert list(df.columns) == ["date", "horizon", "true_adj", "pred_adj", "true_raw", "pred_raw"]

    def test_correct_row_count(self):
        n = 7
        fc = np.ones(n)
        y = np.ones(n)
        dates = np.arange(n)
        bases = np.ones(n)
        df = build_results_dataframe(fc, y, dates, bases, horizon=3)
        assert len(df) == n
        assert (df["horizon"] == 3).all()


class TestExtractSubset:
    def test_pandas_series(self):
        s = pd.Series([10, 20, 30, 40, 50])
        idx = np.array([0, 2, 4])
        result = extract_subset(s, idx)
        np.testing.assert_array_equal(result, np.array([10, 30, 50]))

    def test_numpy_array(self):
        a = np.array([10, 20, 30, 40, 50])
        idx = np.array([1, 3])
        result = extract_subset(a, idx)
        np.testing.assert_array_equal(result, np.array([20, 40]))
