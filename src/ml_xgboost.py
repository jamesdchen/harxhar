# Auto-generated from notebooks/ml_xgboost.ipynb. Do not edit by hand.

"""XGBoost volatility backtest executor."""

import argparse
import json
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.evaluation import apply_duan_smearing
from src.loading import load_raw_data
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
    parser = argparse.ArgumentParser(description="XGBoost walk-forward backtest")
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=500, help="training window in days")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--params-file", default=None, help="JSON file with tuned hyperparams")
    args = parser.parse_args()

    tuned_params = {}
    if args.params_file:
        with open(args.params_file) as f:
            tuned_params = json.load(f)

    train_win_periods = args.train_window * PERIODS_PER_DAY

    df = load_raw_data(args.data_path, allow_missing=True)
    adj_rv, baseline = robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=240)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    df, har_names = generate_har_features(df, target_col="adj_RV")
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
        raise ValueError(f"train_window ({train_win_periods}) >= chunk size ({len(X_chunk)})")

    def model_fn():
        defaults = dict(n_estimators=500, max_depth=5, learning_rate=0.1, tree_method="hist", n_jobs=-1)
        defaults.update(tuned_params)
        return XGBRegressor(**defaults)

    preds = run_backtest(model_fn, X_chunk, y_chunk, train_win=train_win_periods, refit_frequency=20, use_scaling=False)

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

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    results.to_csv(args.output_file, index=False)
    print(f"Saved {len(results)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
