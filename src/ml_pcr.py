"""PCA + Ridge (PCR) walk-forward backtest for volatility forecasting."""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tqdm import tqdm

from src.loading import load_raw_data
from src.transforms import (
    robust_transform,
    resolve_pca_lags,
    generate_raw_lag_features,
    PERIODS_PER_DAY,
)
from src.scaling import RollingRobustScaler


class PCATransform:
    """Thin wrapper around sklearn PCA for the backtest loop."""
    def __init__(self, n_components=5):
        self.pca = PCA(n_components=n_components, svd_solver="randomized")

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)


def run_pcr_backtest(X, y, train_window, n_components=5, refit_frequency=240, alpha=1.0):
    """Walk-forward PCA + Ridge backtest."""
    N, p = X.shape
    n_test = N - train_window
    forecasts = np.empty(n_test, dtype=np.float64)

    scaler = RollingRobustScaler(train_window, p)
    scaler.initialize(X[:train_window])

    pca = PCATransform(n_components=n_components)
    X_buf_scaled = scaler.transform_buffer()
    pca.fit(X_buf_scaled)

    X_buf_pca = pca.transform(X_buf_scaled)
    y_buf = y[:train_window]
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_buf_pca, y_buf)

    steps_since_refit = 0

    for i in tqdm(range(n_test), desc="PCR backtest", leave=False):
        idx = train_window + i
        x_t = X[idx]

        x_scaled = scaler.transform_single(x_t)
        x_pca = pca.transform(x_scaled.reshape(1, -1))
        forecasts[i] = ridge.predict(x_pca)[0]

        scaler.update(x_t)
        steps_since_refit += 1

        if steps_since_refit >= refit_frequency:
            X_buf_scaled = scaler.transform_buffer()
            pca.fit(X_buf_scaled)
            X_buf_pca = pca.transform(X_buf_scaled)
            buf_start = idx + 1 - train_window
            y_buf = y[buf_start : idx + 1]
            ridge.fit(X_buf_pca, y_buf)
            steps_since_refit = 0

    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA + Ridge (PCR) volatility backtest")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=500, help="Training window in days")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output-file", type=str, default="results_pcr.csv")
    parser.add_argument("--n-components", type=int, default=5)
    args = parser.parse_args()

    train_window = args.train_window * PERIODS_PER_DAY

    df = load_raw_data(args.data_path)
    adj_series, baseline = robust_transform(df, "RV", is_target=True)
    df["adj_RV"] = adj_series
    df["baseline"] = baseline

    df, feature_names = generate_raw_lag_features(df, target_col="adj_RV")

    df["target"] = df["adj_RV"].shift(-args.horizon)
    max_lag = resolve_pca_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)
    df = df.dropna(subset=["target"] + feature_names).reset_index(drop=True)

    start = args.start
    end = len(df) if args.end == -1 else args.end
    df_chunk = df.iloc[start:end].reset_index(drop=True)

    X = df_chunk[feature_names].values.astype(np.float64)
    y = df_chunk["target"].values.astype(np.float64)
    dates = df_chunk["t"].values
    baselines_arr = df_chunk["baseline"].values

    forecasts = run_pcr_backtest(X, y, train_window=train_window, n_components=args.n_components, refit_frequency=240)

    y_test = y[train_window:]
    dates_test = dates[train_window:]
    baselines_test = baselines_arr[train_window:]

    smear = np.mean((y_test - forecasts) ** 2)
    pred_raw = (forecasts ** 2 + smear) * baselines_test
    true_raw = (y_test ** 2) * baselines_test

    results = pd.DataFrame({
        "date": dates_test, "horizon": args.horizon,
        "true_adj": y_test, "pred_adj": forecasts,
        "true_raw": true_raw, "pred_raw": pred_raw,
    })

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    results.to_csv(args.output_file, index=False)
    print(f"Saved {len(results)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
