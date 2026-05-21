# Auto-generated from ml_pcr.ipynb. Do not edit by hand.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tqdm import tqdm

from src._template import register_run
from src.evaluation import calculate_metrics
from src.executor import load_and_transform
from src.loading import parse_exog_cols
from src.scaling import RollingRobustScaler
from src.transforms import PERIODS_PER_DAY, generate_raw_lag_features, resolve_pca_lags

PCR_REFIT_FREQUENCY: int = 240


class PCATransform:
    """Thin wrapper around sklearn PCA for the backtest loop."""

    def __init__(self, n_components=5, random_state=42):
        self.pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)


def run_pcr_backtest(X, y, train_window, n_components=5, refit_frequency=240, alpha=1.0, random_state=42):
    """Walk-forward PCA + Ridge backtest."""
    N, p = X.shape
    n_test = N - train_window
    forecasts = np.empty(n_test, dtype=np.float64)

    scaler = RollingRobustScaler(train_window, p)
    scaler.initialize(X[:train_window])

    pca = PCATransform(n_components=n_components, random_state=random_state)
    X_buf_scaled = scaler.transform_buffer()
    pca.fit(X_buf_scaled)

    X_buf_pca = pca.transform(X_buf_scaled)
    y_buf = y[:train_window]
    ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=random_state)
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


def _run_backtest_and_save(
    df: pd.DataFrame,
    feature_names: list[str],
    train_window: int,
    horizon: int,
    start: int,
    end: int,
    output_file: str,
    n_components: int = 5,
    random_state: int = 42,
) -> None:
    """Run PCR backtest on a prepared DataFrame and save results.csv."""
    max_lag = resolve_pca_lags()[-1]

    df["target"] = df["adj_RV"].shift(-horizon)
    df = df.iloc[max_lag:].reset_index(drop=True)
    df = df.dropna(subset=["target"] + feature_names).reset_index(drop=True)

    actual_end = len(df) if end == -1 else end
    df_chunk = df.iloc[start:actual_end].reset_index(drop=True)

    X = df_chunk[feature_names].values.astype(np.float64)
    y = df_chunk["target"].values.astype(np.float64)
    dates = df_chunk["t"].values
    baselines_arr = df_chunk["baseline"].values

    forecasts = run_pcr_backtest(
        X,
        y,
        train_window=train_window,
        n_components=n_components,
        refit_frequency=PCR_REFIT_FREQUENCY,
        random_state=random_state,
    )

    y_test = y[train_window:]
    dates_test = dates[train_window:]
    baselines_test = baselines_arr[train_window:]

    smear = np.mean((y_test - forecasts) ** 2)
    pred_raw = (forecasts**2 + smear) * baselines_test
    true_raw = (y_test**2) * baselines_test

    results = pd.DataFrame(
        {
            "date": dates_test,
            "horizon": horizon,
            "true_adj": y_test,
            "pred_adj": forecasts,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"Saved {len(results)} rows to {output_file}")


@register_run
def run(
    horizon: int = 1,
    train_window: int = 500,
    n_components: int = 5,
    exog_cols: str = "",
    seed: int = 42,
    data_path: str = "all30min",
    output_file: str = "results/pcr/run.json",
    start: int = 0,
    end: int = -1,
) -> dict:
    """PCA + Ridge (PCR) walk-forward volatility backtest -- one task.

    Returns a metrics dict; writes the per-row ``results.csv`` next to
    ``output_file``. Data-prep invariants (formerly ``ExecutorConfig``):
    diurnal-adjusted RV target with no winsorization, leading-edge NaN drop.
    """
    df, adj_exog_cols = load_and_transform(
        data_path,
        parse_exog_cols(exog_cols or None),
        target_use_diurnal=True,
        target_winsor_window=None,
        dropna_with_exog=True,
    )
    df, feature_names = generate_raw_lag_features(df, target_col="adj_RV", exog_cols=adj_exog_cols)

    results_csv = str(Path(output_file).with_name("results.csv"))
    _run_backtest_and_save(
        df,
        feature_names,
        train_window * PERIODS_PER_DAY,
        horizon,
        start,
        end,
        results_csv,
        n_components,
        random_state=seed,
    )
    metrics = calculate_metrics(pd.read_csv(results_csv))
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
