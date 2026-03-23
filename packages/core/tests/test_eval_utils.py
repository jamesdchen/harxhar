"""Tests for src/eval_utils.py — load_all_chunks, parse_config, filter_by_time."""

import numpy as np
import pandas as pd

from harxhar_core.evaluation.aggregation import filter_by_time, load_all_chunks, parse_config

# ---------------------------------------------------------------------------
# Helper to write a simple chunk CSV
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


class TestLoadAllChunks:
    def test_stitches_multiple_csvs(self, tmp_path):
        for i in range(3):
            _write_chunk(tmp_path / f"results_chunk_{i}.csv", n=5, start_date=f"2020-01-0{i + 1}")
        df = load_all_chunks(str(tmp_path))
        assert len(df) == 15
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_empty_directory_returns_empty(self, tmp_path):
        df = load_all_chunks(str(tmp_path))
        assert df.empty

    def test_ignore_suffixes(self, tmp_path):
        _write_chunk(tmp_path / "results_chunk_0_morning.csv", n=3)
        _write_chunk(tmp_path / "results_chunk_0_closing.csv", n=4)
        df = load_all_chunks(str(tmp_path), ignore_suffixes=["morning"])
        assert len(df) == 4

    def test_require_suffixes(self, tmp_path):
        _write_chunk(tmp_path / "results_chunk_0_morning.csv", n=3)
        _write_chunk(tmp_path / "results_chunk_0_closing.csv", n=4)
        df = load_all_chunks(str(tmp_path), require_suffixes=["morning"])
        assert len(df) == 3


class TestParseConfig:
    def test_parses_all_fields(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text("Experiment Name: test_exp\nExperiment ID: 42\nModel Type: ridge\n")
        exp_id, name, model = parse_config(str(tmp_path))
        assert exp_id == 42
        assert name == "test_exp"
        assert model == "ridge"

    def test_missing_config_infers_id_from_path(self, tmp_path):
        # No config.txt — should infer ID from directory name
        exp_dir = tmp_path / "results_7"
        exp_dir.mkdir()
        exp_id, name, model = parse_config(str(exp_dir))
        assert exp_id == 7
        assert name == "Unknown"


class TestFilterByTime:
    def test_filters_to_time_window(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="h")
        df = pd.DataFrame({"val": range(24)}, index=dates)
        result = filter_by_time(df, start_time="09:00:00", end_time="12:00:00")
        # inclusive='left' → hours 9, 10, 11 (12:00 excluded)
        assert len(result) == 3
        assert result.index[0].hour == 9
        assert result.index[-1].hour == 11

    def test_empty_df_passthrough(self):
        df = pd.DataFrame()
        result = filter_by_time(df, start_time="09:00:00", end_time="12:00:00")
        assert result.empty
