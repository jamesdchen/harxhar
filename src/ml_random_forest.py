# Auto-generated from notebooks/ml_random_forest.ipynb. Do not edit by hand.

"""Random Forest volatility backtest executor."""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluation import apply_duan_smearing, calculate_metrics
from src.loading import apply_overnight_fills, load_raw_data, parse_exog_cols
from src.scaling import run_backtest
from src.transforms import (
    PERIODS_PER_DAY,
    add_calendar_features,
    apply_horizon_shift,
    generate_har_features,
    resolve_har_lags,
    robust_transform,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest walk-forward backtest")
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=500, help="training window in days")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--exog-cols", default=None, help="Pipe-separated exog columns, e.g. vix|sentiment")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--params-file", default=None, help="JSON file with tuned hyperparams")
    args = parser.parse_args()

    exog_cols = parse_exog_cols(args.exog_cols)

    tuned_params = {}
    if args.params_file:
        with open(args.params_file) as f:
            tuned_params = json.load(f)

    train_win_periods = args.train_window * PERIODS_PER_DAY

    df = load_raw_data(args.data_path, allow_missing=True)
    if exog_cols:
        apply_overnight_fills(df, exog_cols)
        df = df.dropna(subset=["RV"] + exog_cols).reset_index(drop=True)

    adj_rv, baseline = robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=240)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    adj_exog_cols: list[str] = []
    for col in exog_cols:
        adj_col = f"adj_{col}"
        adj_series, _ = robust_transform(df, col, use_transform=True, use_diurnal=True)
        df[adj_col] = adj_series
        adj_exog_cols.append(adj_col)

    df, har_names = generate_har_features(df, target_col="adj_RV", exog_cols=adj_exog_cols)
    cal_names = add_calendar_features(df)
    feature_names = har_names + cal_names

    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, args.horizon)

    start = args.start
    end = len(X) if args.end == -1 else args.end
    X_chunk, y_chunk = X[start:end], y[start:end]
    dates_chunk = dates.iloc[start:end].reset_index(drop=True)
    baselines_chunk = baselines[start:end]

    if train_win_periods >= len(X_chunk):
        raise ValueError(f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})")

    def model_fn():
        defaults = dict(n_estimators=500, max_depth=10, min_samples_leaf=5, n_jobs=-1)
        defaults.update(tuned_params)
        return RandomForestRegressor(**defaults)

    preds = run_backtest(model_fn, X_chunk, y_chunk, train_win=train_win_periods, refit_frequency=5, use_scaling=False)

    oos_start = train_win_periods
    y_oos = y_chunk[oos_start:]
    dates_oos = dates_chunk.iloc[oos_start:].values
    baselines_oos = baselines_chunk[oos_start:]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": args.horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(args.output_file, index=False)

    metrics = calculate_metrics(results)
    metrics_path = os.path.join(out_dir, f"metrics_chunk_{start}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Saved {len(results)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
