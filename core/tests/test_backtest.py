"""Tests for core/backtest/engine.py functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hpc.chunking import ChunkContext

from core.backtest import (
    apply_duan_smearing,
    build_results_dataframe,
    extract_subset,
)


class TestChunkContextIntegration:
    """Test chunking protocol (hpc.chunking) for backtest index splitting."""

    def test_single_chunk_covers_all_test_indices(self):
        ctx = ChunkContext(chunk_id=0, total_chunks=1, result_dir=Path("."))
        r = ctx.split(range(20, 100))
        assert r == range(20, 100)

    def test_first_chunk_starts_at_train_window(self):
        ctx = ChunkContext(chunk_id=0, total_chunks=2, result_dir=Path("."))
        r = ctx.split(range(20, 100))
        assert r.start == 20

    def test_chunks_cover_all_test_indices(self):
        tw = 10
        total = 4
        all_idx = []
        for c in range(total):
            ctx = ChunkContext(chunk_id=c, total_chunks=total, result_dir=Path("."))
            all_idx.extend(ctx.split(range(tw, 50)))
        assert sorted(all_idx) == list(range(tw, 50))


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
