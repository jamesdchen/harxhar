"""Naive lag-based baseline volatility forecast executor."""

import argparse
import os

import numpy as np
import pandas as pd

from src.loading import load_raw_data
from src.transforms import (
    robust_transform,
    resolve_har_lags,
    generate_har_features,
    apply_horizon_shift,
    PERIODS_PER_DAY,
)
from src.evaluation import apply_duan_smearing


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive baseline backtest")
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=500, help="training window in days (burn-in)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    train_win_periods = args.train_window * PERIODS_PER_DAY

    df = load_raw_data(args.data_path)
    adj_rv, baseline = robust_transform(df, "RV", is_target=True)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    df, feature_names = generate_har_features(df, target_col="adj_RV")
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, args.horizon)

    start = args.start
    end = len(X) if args.end == -1 else args.end
    X_chunk = X[start:end]
    y_chunk = y[start:end]
    dates_chunk = dates.iloc[start:end].reset_index(drop=True)
    baselines_chunk = baselines[start:end]

    if train_win_periods >= len(X_chunk):
        raise ValueError(f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})")

    lag_125_index = feature_names.index("har_ma_125")
    oos_start = train_win_periods
    X_oos = X_chunk[oos_start:]
    y_oos = y_chunk[oos_start:]
    dates_oos = dates_chunk.iloc[oos_start:].values
    baselines_oos = baselines_chunk[oos_start:]
    preds = X_oos[:, lag_125_index]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame({
        "date": dates_oos, "horizon": args.horizon,
        "true_adj": y_oos, "pred_adj": preds,
        "true_raw": true_raw, "pred_raw": pred_raw,
    })

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    results.to_csv(args.output_file, index=False)
    print(f"Saved {len(results)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
