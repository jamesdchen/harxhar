"""Data pipeline: load data and orchestrate feature generation."""

from __future__ import annotations

import numpy as np

from src.core import config
from src.core.config import check_positive
from src.data.loading import load_and_clean_base_data
from src.features.pipeline import (
    generate_lag_features,
    generate_lag_features_segmented,
)

# ---------------------------------------------------------------------------
# Horizon shift utility
# ---------------------------------------------------------------------------


def apply_horizon_shift(X, y, dates, baselines, horizon):
    """
    Shift targets forward by (horizon-1) for direct h-step forecasting.

    At each index t, features X[t] remain unchanged while y[t] becomes the
    value (horizon-1) steps into the future.  Baselines are aligned with the
    target time (needed for Duan smearing); dates are kept at prediction time.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    y : np.ndarray, shape (N,)
    dates : pd.Series of length N
    baselines : np.ndarray of length N
    horizon : int >= 1

    Returns
    -------
    (X, y, dates, baselines) with aligned lengths (N - horizon + 1).
    """
    check_positive(horizon, "horizon")
    if horizon > config.PERIODS_PER_DAY:
        raise ValueError(f"horizon must be <= {config.PERIODS_PER_DAY}, got {horizon}")
    if horizon <= 1:
        return X, y, dates, baselines
    shift = horizon - 1
    return (
        X[:-shift],
        y[shift:],
        dates.iloc[:-shift].reset_index(drop=True),
        baselines[shift:],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_prep_data_strided(
    hparams: dict,
    input_path: str,
    target_segment: str | None = None,
    lag: int | None = None,
) -> tuple | dict:
    """
    Load data and generate lag features for backtesting.

    This function orchestrates two steps:
    1. Load and clean the base data (``load_and_clean_base_data``).
    2. Generate lag features (delegated to ``src.features.pipeline``).

    Parameters
    ----------
    hparams : dict
        Pipeline hyperparameters (feature_type, is_tree, allow_missing, lag_scope).
    input_path : str
        Path to parquet file(s).
    target_segment : str or None
        None          → global mode: returns (X, y, dates, baselines, feature_names).
        'all'         → segmented mode: returns dict of all segments.
        segment name  → segmented mode: returns (X, y, dates, baselines) for one segment.
    lag : int or None
        Maximum lag value. Defaults to config.LAG.
        For HAR: generates a geometric base-5 sequence [1, 5, 25, …] up to lag.
        For raw: generates consecutive lags list(range(1, lag + 1)).
    """
    # Step 1: Load and clean data
    data, cols_to_transform = load_and_clean_base_data(hparams, input_path)
    if data.empty:
        if target_segment == "all":
            return {}
        return np.array([]), np.array([]), [], []

    # Step 2: Generate features
    if target_segment is not None:
        return generate_lag_features_segmented(
            data,
            cols_to_transform,
            hparams,
            target_segment,
            lag=lag,
        )

    return generate_lag_features(data, cols_to_transform, hparams, lag=lag)
