"""Tests for aggregation: parse_config (ML-specific) + core chunk loading & time filtering."""

import numpy as np
import pandas as pd

from core.evaluation.aggregation import filter_by_time, load_all_chunks
from projects.ml.evaluation.aggregation import parse_config

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_chunk(path, n=5, start_date="2020-01-01"):
    dates = pd.date_range(start_date, periods=n, freq="h")
    df = pd.DataFrame(
        {
            "date": dates,
            "true_adj": np.random.rand(n),
            "pred_adj": np.random.rand(n),
            "true_raw": np.random.rand(n) + 0.1,
            "pred_raw": np.random.rand(n) + 0.1,
        }
    )
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# parse_config
# ---------------------------------------------------------------------------


class TestParseConfig:
    def test_valid_config(self, tmp_path):
        (tmp_path / "config.txt").write_text("Experiment Name: my_exp\nExperiment ID: 99\nModel Type: lasso\n")
        exp_id, name, model = parse_config(tmp_path)
        assert exp_id == 99
        assert name == "my_exp"
        assert model == "lasso"

    def test_missing_file_returns_defaults(self, tmp_path):
        exp_dir = tmp_path / "results_5"
        exp_dir.mkdir()
        exp_id, name, model = parse_config(exp_dir)
        # ID inferred from directory name suffix
        assert exp_id == 5
        assert name == "Unknown"
        assert model == "Unknown"

    def test_malformed_file(self, tmp_path):
        (tmp_path / "config.txt").write_text("Experiment ID: not_a_number\ngarbage line\n")
        # ValueError on int() is caught; ID falls back to path inference
        exp_id, name, model = parse_config(tmp_path)
        assert name == "Unknown"
        assert model == "Unknown"


# ---------------------------------------------------------------------------
# load_all_chunks
# ---------------------------------------------------------------------------


class TestLoadAllChunks:
    def test_stitches_multiple_csvs(self, tmp_path):
        for i in range(3):
            _write_chunk(tmp_path / f"results_chunk_{i}.csv", n=5, start_date=f"2020-01-0{i + 1}")
        df = load_all_chunks(str(tmp_path))
        assert len(df) == 15
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_ignore_suffixes(self, tmp_path):
        _write_chunk(tmp_path / "results_chunk_0_am.csv", n=3)
        _write_chunk(tmp_path / "results_chunk_1_pm.csv", n=4)
        df = load_all_chunks(tmp_path, ignore_suffixes=["am"])
        assert len(df) == 4

    def test_require_suffixes(self, tmp_path):
        _write_chunk(tmp_path / "results_chunk_0_am.csv", n=3)
        _write_chunk(tmp_path / "results_chunk_1_pm.csv", n=4)
        df = load_all_chunks(tmp_path, require_suffixes=["pm"])
        assert len(df) == 4

    def test_empty_directory(self, tmp_path):
        df = load_all_chunks(tmp_path)
        assert df.empty


# ---------------------------------------------------------------------------
# filter_by_time
# ---------------------------------------------------------------------------


class TestFilterByTime:
    def test_normal_range(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="h")
        df = pd.DataFrame({"val": range(24)}, index=dates)
        result = filter_by_time(df, start_time="10:00:00", end_time="14:00:00")
        # inclusive='left' → hours 10, 11, 12, 13
        assert len(result) == 4
        assert result.index[0].hour == 10
        assert result.index[-1].hour == 13

    def test_none_bounds_returns_unchanged(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="h")
        df = pd.DataFrame({"val": range(10)}, index=dates)
        result = filter_by_time(df, start_time=None, end_time=None)
        pd.testing.assert_frame_equal(result, df)

    def test_empty_df(self):
        df = pd.DataFrame()
        result = filter_by_time(df)
        assert result.empty
