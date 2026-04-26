# Auto-generated from notebooks/03_evaluation.ipynb. Do not edit by hand.

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


def save_chunk_reduce(df: pd.DataFrame, output_file: str) -> str:
    """Write per-chunk partial metrics next to ``output_file``.

    Produces ``<basename>_reduce.json`` containing counts + sums suitable for
    additive aggregation across chunks. Trial-level QLIKE is
    ``sum(qlike_sum) / sum(qlike_count)``, which is exact because QLIKE is a
    per-row mean.

    Parameters
    ----------
    df : pd.DataFrame
        Results with columns ``true_adj``, ``pred_adj``, ``true_raw``, ``pred_raw``.
    output_file : str
        The chunk's CSV path. Reduce JSON is written to the same basename
        with ``_reduce.json`` suffix.

    Returns
    -------
    str
        Path of the reduce JSON written.
    """
    import json
    import os

    true_raw = np.asarray(df["true_raw"].values, dtype=np.float64)
    pred_raw = np.asarray(df["pred_raw"].values, dtype=np.float64)
    err_adj = np.asarray(df["true_adj"].values, dtype=np.float64) - np.asarray(df["pred_adj"].values, dtype=np.float64)

    mask = (true_raw > 0) & (pred_raw > 0)
    if mask.any():
        ratio = true_raw[mask] / pred_raw[mask]
        qlike_sum = float(np.sum(ratio - np.log(ratio) - 1.0))
    else:
        qlike_sum = 0.0

    partial = {
        "n_samples": int(len(df)),
        "qlike_count": int(mask.sum()),
        "qlike_sum": qlike_sum,
        "mse_sum": float(np.sum(err_adj**2)),
        "mae_sum": float(np.sum(np.abs(err_adj))),
    }
    base, ext = os.path.splitext(output_file)
    reduce_path = (base if ext else output_file) + "_reduce.json"
    with open(reduce_path, "w") as f:
        json.dump(partial, f)
    return reduce_path


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


def mz_regression(y: np.ndarray, yhat: np.ndarray) -> dict:
    """Mincer-Zarnowitz regression: OLS of y on [1, yhat] in raw RV levels.

    Optimal forecasts satisfy alpha=0, beta=1. Returns OLS estimates with
    standard errors and t-statistics for the joint hypothesis.

    Parameters
    ----------
    y : array-like
        Realized variance (raw scale, positive).
    yhat : array-like
        Forecast variance (raw scale, positive).

    Returns
    -------
    dict
        alpha, beta, alpha_se, beta_se, r2, n, t_beta_eq_1, t_alpha_eq_0.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    X = np.column_stack([np.ones_like(yhat), yhat])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = beta_hat[0], beta_hat[1]
    fit = X @ beta_hat
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = len(y)
    sigma2 = ss_res / max(n - 2, 1)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_se": float(se[0]),
        "beta_se": float(se[1]),
        "r2": float(r2),
        "n": int(n),
        "t_beta_eq_1": float((beta - 1.0) / se[1]),
        "t_alpha_eq_0": float(alpha / se[0]),
    }


def qlike_by_slot(df: pd.DataFrame) -> pd.DataFrame:
    """QLIKE stratified by 30-minute intraday slot (0..47).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date (datetime), true_raw, pred_raw.

    Returns
    -------
    pd.DataFrame
        Columns: slot, hour_start, n, qlike, mean_y, mean_yhat (one row per slot).
    """
    d = df.copy()
    d["slot"] = d["date"].dt.hour * 2 + (d["date"].dt.minute >= 30).astype(int)
    rows = []
    for slot, g in d.groupby("slot"):
        true_raw = g["true_raw"].to_numpy()
        pred_raw = g["pred_raw"].to_numpy()
        mask = (true_raw > 0) & (pred_raw > 0)
        if mask.sum() == 0:
            q = float("nan")
        else:
            r = true_raw[mask] / pred_raw[mask]
            q = float(np.mean(r - np.log(r) - 1.0))
        rows.append(
            {
                "slot": int(slot),
                "hour_start": f"{slot // 2:02d}:{(slot % 2) * 30:02d}",
                "n": int(len(g)),
                "qlike": q,
                "mean_y": float(g["true_raw"].mean()),
                "mean_yhat": float(g["pred_raw"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("slot").reset_index(drop=True)


def plot_mz_scatter(
    y: np.ndarray,
    yhat: np.ndarray,
    ax,
    title: str | None = None,
    point_alpha: float = 0.15,
    point_size: float = 2.0,
) -> dict:
    """Mincer-Zarnowitz scatter with mainstream regression-axis convention.

    Plots `ŷ` on horizontal, `y` on vertical (matches OLS `y = α + β·ŷ`).
    ALWAYS draws both the fitted MZ line AND the 45° perfect-forecast
    reference. Returns the MZ regression dict.

    Parameters
    ----------
    y, yhat : array-like
        Realized and forecast RV on raw scale.
    ax : matplotlib Axes
        Target axis. The function does not call savefig.
    title : str | None
        Override title. Default summarises α/β/R²/N.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mz = mz_regression(y, yhat)

    ax.scatter(yhat, y, s=point_size, alpha=point_alpha, rasterized=True, color="steelblue")

    lo = float(min(y.min(), yhat.min()))
    hi = float(max(y.max(), yhat.max()))
    grid = np.array([lo, hi])

    mz_line = mz["alpha"] + mz["beta"] * grid
    ax.plot(grid, mz_line, color="red", lw=1.5, label=f"MZ fit: y = {mz['alpha']:.3g} + {mz['beta']:.3f}·ŷ")
    ax.plot(grid, grid, color="gray", lw=1.0, ls="--", label="45° (perfect forecast)")

    ax.set_xlabel("forecast ŷ (raw RV)")
    ax.set_ylabel("realized y (raw RV)")
    if title is None:
        title = (
            f"MZ: α={mz['alpha']:.3g}, β={mz['beta']:.3f} "
            f"(t_β=1: {mz['t_beta_eq_1']:.2f}), "
            f"R²={mz['r2']:.3f}, N={mz['n']:,}"
        )
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return mz


def plot_y_yhat_timeseries(
    dates,
    y: np.ndarray,
    yhat: np.ndarray,
    ax_raw,
    ax_log=None,
    title: str | None = None,
) -> None:
    """Plot Y vs Ŷ time series. Linear scale on `ax_raw`, optional log on `ax_log`."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ax_raw.plot(dates, y, color="black", lw=0.6, label="realized (y)")
    ax_raw.plot(dates, yhat, color="tab:orange", lw=0.6, alpha=0.85, label="forecast (ŷ)")
    ax_raw.set_ylabel("RV (raw)")
    if title is not None:
        ax_raw.set_title(title)
    ax_raw.legend(loc="upper left")
    ax_raw.grid(True, alpha=0.3)
    if ax_log is not None:
        ax_log.semilogy(dates, y, color="black", lw=0.6, label="realized (y)")
        ax_log.semilogy(dates, yhat, color="tab:orange", lw=0.6, alpha=0.85, label="forecast (ŷ)")
        ax_log.set_ylabel("RV (log)")
        ax_log.set_xlabel("date")
        ax_log.grid(True, which="both", alpha=0.3)


def plot_crash_window(
    df: pd.DataFrame,
    start,
    end,
    ax_raw,
    ax_log=None,
    title: str | None = None,
) -> None:
    """Plot Y vs Ŷ for a crash-window time range.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date (datetime), true_raw, pred_raw.
    start, end : str | pd.Timestamp
        Inclusive crash-window bounds.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sub = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
    if sub.empty:
        if ax_raw is not None:
            ax_raw.text(
                0.5,
                0.5,
                f"No data in {start.date()} – {end.date()}",
                ha="center",
                va="center",
                transform=ax_raw.transAxes,
            )
        return
    plot_y_yhat_timeseries(
        sub["date"],
        sub["true_raw"].to_numpy(),
        sub["pred_raw"].to_numpy(),
        ax_raw,
        ax_log,
        title=(title or f"{start.date()} – {end.date()}  (N={len(sub):,})"),
    )


def plot_qlike_by_slot(
    slot_df: pd.DataFrame,
    ax,
    global_qlike: float | None = None,
    title: str | None = None,
) -> None:
    """Bar chart of per-slot QLIKE; optional horizontal line for the global QLIKE."""
    ax.bar(slot_df["slot"], slot_df["qlike"], width=0.8, color="steelblue")
    if global_qlike is not None:
        ax.axhline(global_qlike, color="red", lw=1, ls="--", label=f"global QLIKE = {global_qlike:.4f}")
        ax.legend(loc="upper right")
    ax.set_xlabel("30-min intraday slot (0 = 00:00, 47 = 23:30)")
    ax.set_ylabel("QLIKE")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
