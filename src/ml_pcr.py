"""PCA + Ridge (PCR) walk-forward backtest for volatility forecasting.

Standalone module — no imports from core/ or projects/.
Uses: numpy, pandas, argparse, os, tqdm, sklearn.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tqdm import tqdm

from evaluation import apply_duan_smearing, calculate_metrics
from src.loading import load_raw_data
from src.scaling import RollingRobustScaler
from src.transforms import robust_transform

# ── Constants ──────────────────────────────────────────────────────────
PERIODS_PER_DAY = 48


# ── Log-spaced lags ───────────────────────────────────────────────────


def resolve_pca_lags(max_lag: int = 3125, num_points: int = 20) -> list[int]:
    """Generate log-spaced lag indices from 1 to *max_lag*."""
    raw = np.geomspace(1, max_lag, num=num_points)
    return sorted(set(int(round(v)) for v in raw))


def generate_raw_lag_features(
    df: pd.DataFrame,
    target_col: str = "adj_RV",
    max_lag: int = 3125,
) -> tuple[pd.DataFrame, list[str]]:
    """Create shifted-lag columns for each log-spaced lag."""
    lags = resolve_pca_lags(max_lag)
    features: dict[str, pd.Series] = {}
    feature_names: list[str] = []
    for lag in lags:
        name = f"{target_col}_lag_{lag}"
        features[name] = df[target_col].shift(lag)
        feature_names.append(name)
    feat_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feat_df], axis=1), feature_names


# ── PCA transform wrapper ────────────────────────────────────────────


class PCATransform:
    """Thin wrapper around sklearn PCA for the backtest loop."""

    def __init__(self, n_components: int = 5):
        self.pca = PCA(n_components=n_components, svd_solver="randomized")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCATransform":
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)


# ── Walk-forward PCR backtest ─────────────────────────────────────────


def run_pcr_backtest(
    X: np.ndarray,
    y: np.ndarray,
    train_window: int,
    n_components: int = 5,
    refit_frequency: int = 240,
    alpha: float = 1.0,
) -> np.ndarray:
    """Walk-forward PCA + Ridge backtest.

    Parameters
    ----------
    X : (N, p) feature matrix (raw lag features, already winsorized/transformed).
    y : (N,) target vector.
    train_window : int
        Number of observations in the rolling training window.
    n_components : int
        PCA dimensionality.
    refit_frequency : int
        Re-fit PCA every this many steps (Ridge is also re-fit at PCA refit).
    alpha : float
        Ridge regularisation strength.

    Returns
    -------
    forecasts : (N - train_window,) array of predictions.
    """
    N, p = X.shape
    n_test = N - train_window
    forecasts = np.empty(n_test, dtype=np.float64)

    # ── Initialise scaler on first window ──
    scaler = RollingRobustScaler(train_window)
    scaler.initialize(X[:train_window])

    # ── Fit PCA on scaled buffer ──
    pca = PCATransform(n_components=n_components)
    X_buf_scaled = scaler.transform_buffer()
    pca.fit(X_buf_scaled)

    # ── Fit Ridge on PCA-transformed buffer ──
    X_buf_pca = pca.transform(X_buf_scaled)
    y_buf = y[:train_window]
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_buf_pca, y_buf)

    steps_since_refit = 0

    for i in tqdm(range(n_test), desc="PCR backtest", leave=False):
        idx = train_window + i
        x_t = X[idx]

        # Scale and PCA-transform the new observation
        x_scaled = scaler.transform_single(x_t)
        x_pca = pca.transform(x_scaled.reshape(1, -1))

        # Predict
        forecasts[i] = ridge.predict(x_pca)[0]

        # Update scaler buffer
        scaler.update(x_t)
        steps_since_refit += 1

        # Periodic PCA + Ridge refit
        if steps_since_refit >= refit_frequency:
            X_buf_scaled = scaler.transform_buffer()
            pca.fit(X_buf_scaled)
            X_buf_pca = pca.transform(X_buf_scaled)

            # Reconstruct y buffer in correct order
            buf_start = idx + 1 - train_window
            y_buf = y[buf_start : idx + 1]
            ridge.fit(X_buf_pca, y_buf)
            steps_since_refit = 0

    return forecasts


# ── CLI entry point ───────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA + Ridge (PCR) volatility backtest")
    parser.add_argument("--data-path", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in periods")
    parser.add_argument("--train-window", type=int, default=500, help="Training window in days")
    parser.add_argument("--start", type=int, default=0, help="Start index for slicing")
    parser.add_argument("--end", type=int, default=-1, help="End index for slicing (-1 means all)")
    parser.add_argument("--output-file", type=str, default="results_pcr.csv", help="Output CSV path")
    parser.add_argument("--n-components", type=int, default=5, help="Number of PCA components")
    args = parser.parse_args()

    train_window = args.train_window * PERIODS_PER_DAY

    # ── Load data ─────────────────────────────────────────────────────
    df = load_raw_data(args.data_path)

    # ── Transform target ──────────────────────────────────────────────
    adj_series, baseline = robust_transform(df, "RV", is_target=True)
    df["adj_RV"] = adj_series
    df["baseline"] = baseline

    # ── Generate raw lag features ─────────────────────────────────────
    df, feature_names = generate_raw_lag_features(df, target_col="adj_RV")

    # ── Horizon shift ─────────────────────────────────────────────────
    df["target"] = df["adj_RV"].shift(-args.horizon)

    # ── Drop NaN rows (from lags + horizon shift) ─────────────────────
    max_lag = resolve_pca_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)
    df = df.dropna(subset=["target"] + feature_names).reset_index(drop=True)

    # ── Slice selection ─────────────────────────────────────────────────
    start = args.start
    end = len(df) if args.end == -1 else args.end
    df_chunk = df.iloc[start:end].reset_index(drop=True)

    # ── Prepare arrays ────────────────────────────────────────────────
    X = df_chunk[feature_names].values.astype(np.float64)
    y = df_chunk["target"].values.astype(np.float64)
    dates = df_chunk["t"].values
    baselines = df_chunk["baseline"].values

    # ── Run backtest ──────────────────────────────────────────────────
    forecasts = run_pcr_backtest(
        X,
        y,
        train_window=train_window,
        n_components=args.n_components,
        refit_frequency=240,
    )

    # ── Duan smearing + save ──────────────────────────────────────────
    y_test = y[train_window:]
    dates_test = dates[train_window:]
    baselines_test = baselines[train_window:]

    pred_raw, true_raw = apply_duan_smearing(forecasts, y_test, baselines_test)

    results = pd.DataFrame(
        {
            "date": dates_test,
            "horizon": args.horizon,
            "true_adj": y_test,
            "pred_adj": forecasts,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(args.output_file, index=False)

    metrics = calculate_metrics(results)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Saved {len(results)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
