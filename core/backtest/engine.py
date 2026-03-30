from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.core.config import check_backtest_inputs, check_finite, check_positive
from core.models import BaseModel


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


def apply_duan_smearing(
    forecasts: np.ndarray, y_true: np.ndarray, baselines: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Duan's smearing estimator to convert from adjusted to raw space."""
    check_finite(forecasts, "forecasts")
    check_finite(y_true, "y_true")
    check_finite(baselines, "baselines")
    smear = np.mean((y_true - forecasts) ** 2)
    pred_raw = (forecasts**2 + smear) * baselines
    true_raw = (y_true**2) * baselines
    return pred_raw, true_raw


def build_results_dataframe(
    forecasts: np.ndarray,
    y_subset: np.ndarray,
    dates_subset: np.ndarray,
    baselines_subset: np.ndarray,
    horizon: int = 1,
) -> pd.DataFrame:
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


def extract_subset(data: pd.Series | pd.DataFrame | np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Extract subset from pandas Series/DataFrame or numpy array."""
    return np.asarray(data.iloc[indices]) if hasattr(data, "iloc") else data[indices]


def save_chunk_results(
    output_file: str | Path,
    forecasts: np.ndarray,
    indices: np.ndarray,
    train_window: int,
    y_true: np.ndarray,
    dates: pd.Series | np.ndarray,
    baselines: np.ndarray,
    horizon: int = 1,
) -> np.ndarray:
    """Saves predictions and reconstructs raw space values for the primary model only."""
    y_subset = y_true[indices]
    base_subset = baselines[indices]
    dates_subset = extract_subset(dates, indices)

    df = build_results_dataframe(forecasts, y_subset, dates_subset, base_subset, horizon=horizon)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return dates_subset
