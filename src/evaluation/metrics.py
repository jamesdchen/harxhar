"""Forecast evaluation metrics: MSE, MAE, QLIKE, and out-of-sample R²."""

import numpy as np
import pandas as pd

from src.core.config import WINSOR_LOWER_Q, WINSOR_UPPER_Q
from src.core.log import get_logger

logger = get_logger(__name__)


def winsorize_series(
    s: np.ndarray | pd.Series,
    lower_q: float = WINSOR_LOWER_Q,
    upper_q: float = WINSOR_UPPER_Q,
) -> np.ndarray:
    """Clip values to [lower_q, upper_q] percentile bounds."""
    arr = np.asarray(s, dtype=float)
    lo, hi = np.nanpercentile(arr, [lower_q * 100, upper_q * 100])
    return np.clip(arr, lo, hi)


def calculate_global_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculates MSE, MAE, and QLIKE — both raw and winsorized variants."""
    metrics: dict[str, float] = {"n_samples": float(len(df))}

    # 1. Adjusted Scale Metrics
    if "true_adj" in df.columns and "pred_adj" in df.columns:
        errors_sq = (df["true_adj"] - df["pred_adj"]) ** 2
        errors_abs = np.abs(df["true_adj"] - df["pred_adj"])

        metrics["mse"] = np.mean(errors_sq)
        metrics["mae"] = np.mean(errors_abs)

        # Winsorized variants — clip the error distribution then average
        metrics["w_mse"] = float(np.mean(winsorize_series(errors_sq.values)))
        metrics["w_mae"] = float(np.mean(winsorize_series(errors_abs.values)))

    # 2. Raw Scale Metrics
    if "true_raw" in df.columns and "pred_raw" in df.columns:
        mask_raw = (df["true_raw"] > 0) & (df["pred_raw"] > 0)
        if mask_raw.sum() > 0:
            vol_true = df.loc[mask_raw, "true_raw"]
            vol_pred = df.loc[mask_raw, "pred_raw"]
            qlike_vals = (vol_true / vol_pred) - np.log(vol_true / vol_pred) - 1

            metrics["qlike"] = np.mean(qlike_vals)
            metrics["w_qlike"] = float(np.mean(winsorize_series(qlike_vals.values)))
        else:
            metrics["qlike"] = np.nan
            metrics["w_qlike"] = np.nan

    return metrics


def calculate_baseline_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the baseline model and computes relative deltas and OOS R2.
    This keeps the comparative logic completely isolated from the file processing.
    """
    baseline_mask = (summary_df["exp_id"] == 0) | (
        summary_df["experiment_name"].str.lower().isin(["baseline", "naive_baseline"])
    )
    baseline_df = summary_df[baseline_mask]

    delta_cols = ["delta_mse", "delta_mae", "delta_qlike", "oos_r2"]
    w_delta_cols = ["delta_w_mse", "delta_w_mae", "delta_w_qlike", "w_oos_r2"]

    if baseline_df.empty:
        logger.warning("No baseline experiment found. Deltas and OOS R2 will be NaN.")
        for col in delta_cols + w_delta_cols:
            summary_df[col] = np.nan
        return summary_df

    # Build a (segment, horizon) -> baseline metrics lookup via merge (vectorized)
    group_cols = ["segment", "horizon"] if "horizon" in summary_df.columns else ["segment"]
    metric_keys = ["mse", "mae", "qlike", "w_mse", "w_mae", "w_qlike"]
    available_keys = [k for k in metric_keys if k in baseline_df.columns]
    baseline_lookup = (
        baseline_df.groupby(group_cols)[available_keys].first().rename(columns={k: f"b_{k}" for k in available_keys})
    )
    summary_df = summary_df.merge(baseline_lookup, on=group_cols, how="left")

    summary_df["delta_mse"] = summary_df["mse"] - summary_df["b_mse"]
    summary_df["delta_mae"] = summary_df["mae"] - summary_df["b_mae"]
    summary_df["delta_qlike"] = summary_df["qlike"] - summary_df["b_qlike"]
    summary_df["oos_r2"] = np.where(
        summary_df["b_mse"] > 0,
        1.0 - summary_df["mse"] / summary_df["b_mse"],
        np.nan,
    )

    # Winsorized deltas
    if "b_w_mse" in summary_df.columns:
        summary_df["delta_w_mse"] = summary_df["w_mse"] - summary_df["b_w_mse"]
        summary_df["delta_w_mae"] = summary_df["w_mae"] - summary_df["b_w_mae"]
        summary_df["delta_w_qlike"] = summary_df["w_qlike"] - summary_df["b_w_qlike"]
        summary_df["w_oos_r2"] = np.where(
            summary_df["b_w_mse"] > 0,
            1.0 - summary_df["w_mse"] / summary_df["b_w_mse"],
            np.nan,
        )

    b_cols = [c for c in summary_df.columns if c.startswith("b_")]
    summary_df.drop(columns=b_cols, inplace=True)
    return summary_df
