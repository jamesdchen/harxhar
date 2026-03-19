import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)


def calculate_global_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculates MSE, MAE, and QLIKE (both filtered and non-filtered)."""
    metrics = {"n_samples": len(df)}

    # 1. Adjusted Scale Metrics
    if "true_adj" in df.columns and "pred_adj" in df.columns:
        metrics["mse"] = np.mean((df["true_adj"] - df["pred_adj"]) ** 2)
        metrics["mae"] = np.mean(np.abs(df["true_adj"] - df["pred_adj"]))

    # 2. Raw Scale Metrics
    if "true_raw" in df.columns and "pred_raw" in df.columns:
        mask_raw = (df["true_raw"] > 0) & (df["pred_raw"] > 0)
        if mask_raw.sum() > 0:
            vol_true = df.loc[mask_raw, "true_raw"]
            vol_pred = df.loc[mask_raw, "pred_raw"]
            metrics["qlike"] = np.mean((vol_true / vol_pred) - np.log(vol_true / vol_pred) - 1)
        else:
            metrics["qlike"] = np.nan

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

    if baseline_df.empty:
        logger.warning("No baseline experiment found. Deltas and OOS R2 will be NaN.")
        for col in ["delta_mse", "delta_mae", "delta_qlike", "oos_r2"]:
            summary_df[col] = np.nan
        return summary_df

    # Build a segment -> baseline metrics lookup via merge (vectorized)
    baseline_lookup = (
        baseline_df.groupby("segment")[["mse", "mae", "qlike"]]
        .first()
        .rename(columns={"mse": "b_mse", "mae": "b_mae", "qlike": "b_qlike"})
    )
    summary_df = summary_df.merge(baseline_lookup, on="segment", how="left")

    summary_df["delta_mse"] = summary_df["mse"] - summary_df["b_mse"]
    summary_df["delta_mae"] = summary_df["mae"] - summary_df["b_mae"]
    summary_df["delta_qlike"] = summary_df["qlike"] - summary_df["b_qlike"]
    summary_df["oos_r2"] = np.where(
        summary_df["b_mse"] > 0,
        1.0 - summary_df["mse"] / summary_df["b_mse"],
        np.nan,
    )

    summary_df.drop(columns=["b_mse", "b_mae", "b_qlike"], inplace=True)
    return summary_df
