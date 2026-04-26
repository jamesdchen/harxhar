"""Smoke tests for MZ + standardized plot helpers added to src.evaluation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    mz_regression,
    qlike_by_slot,
    plot_mz_scatter,
    plot_y_yhat_timeseries,
    plot_crash_window,
    plot_qlike_by_slot,
)


@pytest.fixture
def synthetic_predictions():
    rng = np.random.default_rng(42)
    y = np.abs(rng.normal(loc=1.0, scale=0.5, size=500))
    yhat = 0.8 * y + 0.05 + rng.normal(scale=0.05, size=500)
    return y, yhat


@pytest.fixture
def synthetic_dataframe(synthetic_predictions):
    y, yhat = synthetic_predictions
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(y), freq="30min"),
            "true_raw": y,
            "pred_raw": yhat,
        }
    )


def test_mz_regression_recovers_known_slope(synthetic_predictions):
    """Forecast was built as yhat = 0.8 y + 0.05 + noise → MZ slope ≈ 1/0.8 = 1.25."""
    y, yhat = synthetic_predictions
    mz = mz_regression(y, yhat)
    # Allow loose tolerance — 500 noisy samples
    assert 1.10 < mz["beta"] < 1.40
    assert -0.15 < mz["alpha"] < 0.05
    assert mz["r2"] > 0.95
    assert mz["n"] == 500


def test_plot_mz_scatter_draws_both_lines(synthetic_predictions):
    """The bug fix: MZ scatter must draw both the fitted MZ line AND the 45° reference."""
    y, yhat = synthetic_predictions
    fig, ax = plt.subplots()
    stats = plot_mz_scatter(y, yhat, ax)
    labels = [line.get_label() for line in ax.get_lines()]
    has_mz_fit = any("MZ fit" in label for label in labels)
    has_45 = any("45" in label for label in labels)
    assert has_mz_fit, f"plot_mz_scatter is missing the MZ fit line; labels={labels}"
    assert has_45, f"plot_mz_scatter is missing the 45° reference; labels={labels}"
    # Returned stats must equal the standalone regression
    assert stats == mz_regression(y, yhat)
    plt.close(fig)


def test_plot_mz_scatter_axes_are_mainstream(synthetic_predictions):
    """Forecast on horizontal, realized on vertical (matches y = α + β·ŷ)."""
    y, yhat = synthetic_predictions
    fig, ax = plt.subplots()
    plot_mz_scatter(y, yhat, ax)
    assert "forecast" in ax.get_xlabel().lower() or "ŷ" in ax.get_xlabel()
    assert "realized" in ax.get_ylabel().lower() or "y" in ax.get_ylabel()
    plt.close(fig)


def test_qlike_by_slot_returns_expected_columns(synthetic_dataframe):
    df = qlike_by_slot(synthetic_dataframe)
    assert {"slot", "hour_start", "n", "qlike", "mean_y", "mean_yhat"} <= set(df.columns)
    assert df["slot"].between(0, 47).all()


def test_plot_y_yhat_timeseries_renders_two_axes(synthetic_dataframe):
    fig, (ax_raw, ax_log) = plt.subplots(2, 1)
    plot_y_yhat_timeseries(
        synthetic_dataframe["date"],
        synthetic_dataframe["true_raw"].to_numpy(),
        synthetic_dataframe["pred_raw"].to_numpy(),
        ax_raw,
        ax_log,
    )
    assert len(ax_raw.get_lines()) == 2
    assert len(ax_log.get_lines()) == 2
    plt.close(fig)


def test_plot_crash_window_handles_empty_range(synthetic_dataframe):
    """Out-of-range crash window emits a placeholder, not a crash."""
    fig, ax = plt.subplots()
    plot_crash_window(synthetic_dataframe, "1990-01-01", "1990-12-31", ax)
    plt.close(fig)


def test_plot_qlike_by_slot_renders(synthetic_dataframe):
    slot_df = qlike_by_slot(synthetic_dataframe)
    fig, ax = plt.subplots()
    plot_qlike_by_slot(slot_df, ax, global_qlike=0.1)
    plt.close(fig)
