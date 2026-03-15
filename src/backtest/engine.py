from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import check_backtest_inputs, check_finite, check_positive
from src.models import BaseModel


def run_backtest_agnostic(
    model: BaseModel,
    indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_win_periods: int,
    save_coefs: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    A truly model-agnostic walk-forward backtester.
    """
    check_positive(train_win_periods, "train_win_periods")
    check_backtest_inputs(X, y, indices)

    first_test_idx = indices[0]
    if first_test_idx < train_win_periods:
        raise ValueError("Not enough history for the requested training window.")

    # 1. Provide the initial burn-in history to the model
    start_hist = first_test_idx - train_win_periods
    X_init = X[start_hist:first_test_idx]
    y_init = y[start_hist:first_test_idx]

    # Model handles its own scaling, buffering, and initial fitting
    model.initialize(X_init, y_init)

    n_preds = len(indices)
    preds = np.zeros(n_preds)
    coef_history = None

    if save_coefs:
        init_coefs = model.get_coefs()
        if init_coefs is not None:
            coef_history = np.zeros((n_preds, len(init_coefs)))

    # 2. Walk-Forward Loop
    for i, t_idx in enumerate(tqdm(indices, desc="Backtesting")):
        x_target = X[t_idx]

        # A. Capture coefficients (model fit on history up to t-1)
        if coef_history is not None:
            coef_history[i, :] = model.get_coefs()

        # B. Predict step t
        preds[i] = model.predict(x_target)

        # C. Observe realized y at step t and let the model update itself
        y_realized = y[t_idx]
        model.update(x_target, y_realized)

    return preds, coef_history


def get_chunk_indices_strided(X_np, train_window_size, chunk_id, total_chunks):
    """Calculates indices for chunked evaluation."""
    num_samples = X_np.shape[0]
    valid_test_start = train_window_size
    if valid_test_start >= num_samples:
        return np.array([])
    test_indices = np.arange(valid_test_start, num_samples)
    chunk_indices_list = np.array_split(test_indices, total_chunks)
    if chunk_id >= len(chunk_indices_list):
        return np.array([])
    return chunk_indices_list[chunk_id]


def apply_duan_smearing(forecasts, y_true, baselines):
    """Apply Duan's smearing estimator to convert from adjusted to raw space."""
    check_finite(forecasts, "forecasts")
    check_finite(y_true, "y_true")
    check_finite(baselines, "baselines")
    smear = np.mean((y_true - forecasts) ** 2)
    pred_raw = (forecasts**2 + smear) * baselines
    true_raw = (y_true**2) * baselines
    return pred_raw, true_raw


def _build_results_dataframe(forecasts, y_subset, dates_subset, baselines_subset, horizon=1):
    """Build a results DataFrame with adjusted and raw-space columns."""
    pred_raw, true_raw = apply_duan_smearing(forecasts, y_subset, baselines_subset)
    return pd.DataFrame(
        {
            "date": dates_subset,
            "horizon": horizon,
            "true_adj": y_subset,
            "pred_adj": forecasts,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )


def _extract_subset(data, indices):
    """Extract subset from pandas Series/DataFrame or numpy array."""
    return data.iloc[indices].values if hasattr(data, "iloc") else data[indices]


def save_chunk_results(output_file, forecasts, indices, train_window, y_true, dates, baselines, horizon=1):
    """Saves predictions and reconstructs raw space values for the primary model only."""
    y_subset = y_true[indices]
    base_subset = baselines[indices]
    dates_subset = _extract_subset(dates, indices)

    df = _build_results_dataframe(forecasts, y_subset, dates_subset, base_subset, horizon=horizon)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return dates_subset
