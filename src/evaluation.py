"""Evaluation metrics and Duan smearing for volatility forecasts.

Standalone module — no imports from core/ or projects/.
"""

import numpy as np
import pandas as pd


def winsorize_array(arr: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95) -> np.ndarray:
    """Clip values to [lower_q, upper_q] percentile bounds."""
    lo = np.percentile(arr, lower_q * 100)
    hi = np.percentile(arr, upper_q * 100)
    return np.clip(arr, lo, hi)


def apply_duan_smearing(
    forecasts: np.ndarray,
    y_true: np.ndarray,
    baselines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Duan smearing correction to convert adjusted-scale forecasts to raw scale.

    Parameters
    ----------
    forecasts : array-like
        Model predictions on adjusted (sqrt / log) scale.
    y_true : array-like
        True values on adjusted scale.
    baselines : array-like
        Baseline volatility used to scale back to raw units.

    Returns
    -------
    pred_raw : np.ndarray
        Smearing-corrected predictions on raw scale.
    true_raw : np.ndarray
        True values on raw scale.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    baselines = np.asarray(baselines, dtype=np.float64)

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
    """Build a tidy results DataFrame with adjusted and raw-scale columns.

    Parameters
    ----------
    forecasts : array-like
        Model predictions (adjusted scale).
    y_subset : array-like
        True targets (adjusted scale).
    dates_subset : array-like
        Corresponding dates.
    baselines_subset : array-like
        Baseline volatility for raw-scale conversion.
    horizon : int
        Forecast horizon label.

    Returns
    -------
    pd.DataFrame
        Columns: date, horizon, true_adj, pred_adj, true_raw, pred_raw.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    y_subset = np.asarray(y_subset, dtype=np.float64)
    baselines_subset = np.asarray(baselines_subset, dtype=np.float64)

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


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Compute evaluation metrics from a results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: true_adj, pred_adj, true_raw, pred_raw.

    Returns
    -------
    dict
        Keys: mse, mae, w_mse, w_mae, qlike, w_qlike, n_samples.
    """
    true_adj = df["true_adj"].values
    pred_adj = df["pred_adj"].values
    true_raw = df["true_raw"].values
    pred_raw = df["pred_raw"].values

    # --- Adjusted-scale errors ---
    errors = true_adj - pred_adj
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))

    # Winsorized variants
    w_errors = winsorize_array(errors)
    w_mse = float(np.mean(w_errors**2))
    w_mae = float(np.mean(np.abs(w_errors)))

    # --- QLIKE (raw scale) ---
    mask = (true_raw > 0) & (pred_raw > 0)
    if mask.sum() > 0:
        ratio = true_raw[mask] / pred_raw[mask]
        qlike_vals = ratio - np.log(ratio) - 1.0
        qlike = float(np.mean(qlike_vals))
        w_qlike = float(np.mean(winsorize_array(qlike_vals)))
    else:
        qlike = np.nan
        w_qlike = np.nan

    return {
        "mse": mse,
        "mae": mae,
        "w_mse": w_mse,
        "w_mae": w_mae,
        "qlike": qlike,
        "w_qlike": w_qlike,
        "n_samples": len(df),
    }
