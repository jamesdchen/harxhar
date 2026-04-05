"""Naive lag-based baseline volatility forecast executor.

Predicts y[t] = har_ma_125[t] (the 125-period HAR rolling mean).
Same CLI interface as ml_ridge.py.  No imports from core/ or projects/.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from evaluation import apply_duan_smearing, calculate_metrics
from src.loading import load_raw_data
from src.transforms import robust_transform

# ── Constants ─────────────────────────────────────────────────────────────
PERIODS_PER_DAY = 48


# ── HAR lag features (same as ml_ridge) ───────────────────────────────────


def resolve_har_lags(max_lag: int = 3125) -> list[int]:
    seq, v = [], 1
    while v <= max_lag:
        seq.append(v)
        v *= 5
    return seq


def generate_har_features(df: pd.DataFrame, target_col: str = "adj_RV") -> tuple[pd.DataFrame, list[str]]:
    lags = resolve_har_lags()
    features: dict[str, pd.Series] = {}
    feature_names: list[str] = []
    for lag in lags:
        name = f"har_ma_{lag}"
        features[name] = df[target_col].rolling(window=lag, min_periods=1).mean().shift(1)
        feature_names.append(name)
    feat_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feat_df], axis=1), feature_names


# ── Horizon shift ─────────────────────────────────────────────────────────


def apply_horizon_shift(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    baselines: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    if horizon <= 1:
        return X, y, dates, baselines
    shift = horizon - 1
    return (
        X[:-shift],
        y[shift:],
        dates.iloc[:-shift].reset_index(drop=True),
        baselines[shift:],
    )


# ── CLI ───────────────────────────────────────────────────────────────────


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

    # 1. Load data
    df = load_raw_data(args.data_path)

    # 2. Robust transform on RV
    adj_rv, baseline = robust_transform(df, "RV", is_target=True)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    # 3. HAR features
    df, feature_names = generate_har_features(df, target_col="adj_RV")

    # 4. Drop initial NaN rows
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    # 5. Extract numpy arrays
    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    # 6. Horizon shift
    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, args.horizon)

    # 7. Slice
    start = args.start
    end = len(X) if args.end == -1 else args.end

    X_chunk = X[start:end]
    y_chunk = y[start:end]
    dates_chunk = dates.iloc[start:end].reset_index(drop=True)
    baselines_chunk = baselines[start:end]

    if train_win_periods >= len(X_chunk):
        raise ValueError(f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})")

    # 8. Naive prediction: y_pred[t] = har_ma_125[t]
    lag_125_index = feature_names.index("har_ma_125")

    oos_start = train_win_periods
    X_oos = X_chunk[oos_start:]
    y_oos = y_chunk[oos_start:]
    dates_oos = dates_chunk.iloc[oos_start:].values
    baselines_oos = baselines_chunk[oos_start:]

    preds = X_oos[:, lag_125_index]

    # 9. Duan smearing + save
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
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Saved {len(results)} rows → {args.output_file}")


if __name__ == "__main__":
    main()
