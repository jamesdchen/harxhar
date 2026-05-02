# Auto-generated from notebooks/ml_pcr.ipynb. Do not edit by hand.

"""PCA + Ridge (PCR) walk-forward backtest for volatility forecasting."""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tqdm import tqdm

from src.executor import CONFIGS, load_and_transform
from src.loading import parse_exog_cols
from src.scaling import RollingRobustScaler
from src.transforms import (
    PERIODS_PER_DAY,
    SEGMENT_DEFINITIONS,
    compute_segment_train_window,
    generate_raw_lag_features,
    resolve_pca_lags,
    slice_to_segment,
)


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
    """Run PCR backtest on a prepared DataFrame and save results."""
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
        refit_frequency=CONFIGS["pcr"].refit_frequency,
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


def compute(args) -> None:
    exog_cols = parse_exog_cols(args.exog_cols)

    # --- Load and transform (shared with executor.run_executor) ---
    df, adj_exog_cols = load_and_transform(
        args.data_path,
        exog_cols,
        target_use_diurnal=CONFIGS["pcr"].target_use_diurnal,
        target_winsor_window=CONFIGS["pcr"].target_winsor_window,
        dropna_with_exog=CONFIGS["pcr"].dropna_with_exog,
    )

    # --- No segment: global backtest ---
    if args.segment is None:
        train_window = args.train_window * PERIODS_PER_DAY
        df, feature_names = generate_raw_lag_features(df, target_col="adj_RV", exog_cols=adj_exog_cols)
        _run_backtest_and_save(
            df,
            feature_names,
            train_window,
            args.horizon,
            args.start,
            args.end,
            args.output_file,
            args.n_components,
            random_state=args.seed,
        )
        return

    # --- Segmented backtest ---
    segments = list(SEGMENT_DEFINITIONS) if args.segment == "all" else [args.segment]

    if args.lag_scope == "global":
        df, feature_names = generate_raw_lag_features(df, target_col="adj_RV", exog_cols=adj_exog_cols)

    for seg_name in segments:
        seg_df = slice_to_segment(df, seg_name)
        if seg_df.empty:
            print(f"No data for segment '{seg_name}'. Skipping.")
            continue

        if args.lag_scope == "intra":
            seg_df, feature_names = generate_raw_lag_features(seg_df, target_col="adj_RV", exog_cols=adj_exog_cols)

        train_window = compute_segment_train_window(seg_df["t"], args.train_window)

        base, ext = os.path.splitext(args.output_file)
        seg_output = f"{base}_{seg_name}{ext}"

        print(f"{'=' * 20} SEGMENT: {seg_name.upper()} {'=' * 20}")
        print(f"Window: {train_window} periods ({args.train_window} days)")
        _run_backtest_and_save(
            seg_df,
            feature_names,
            train_window,
            args.horizon,
            args.start,
            args.end,
            seg_output,
            args.n_components,
            random_state=args.seed,
        )
