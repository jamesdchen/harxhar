# Auto-generated from notebooks/ml_ridge.ipynb. Do not edit by hand.

"""Ridge regression volatility backtest executor."""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.evaluation import apply_duan_smearing
from src.loading import apply_overnight_fills, load_raw_data, parse_exog_cols
from src.scaling import run_backtest
from src.transforms import (
    PERIODS_PER_DAY,
    SEGMENT_CHOICES,
    SEGMENT_DEFINITIONS,
    apply_horizon_shift,
    compute_segment_train_window,
    generate_har_features,
    resolve_har_lags,
    robust_transform,
    slice_to_segment,
)


def _run_backtest_and_save(
    df: pd.DataFrame,
    feature_names: list[str],
    train_win_periods: int,
    horizon: int,
    start: int,
    end: int,
    output_file: str,
) -> None:
    """Run Ridge backtest on a prepared DataFrame and save results."""
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon)

    actual_end = len(X) if end == -1 else end
    X_chunk = X[start:actual_end]
    y_chunk = y[start:actual_end]
    dates_chunk = dates.iloc[start:actual_end].reset_index(drop=True)
    baselines_chunk = baselines[start:actual_end]

    if train_win_periods >= len(X_chunk):
        raise ValueError(f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})")

    model_fn = lambda: Ridge(alpha=1.0)
    preds = run_backtest(
        model_fn,
        X_chunk,
        y_chunk,
        train_win=train_win_periods,
        refit_frequency=1,
        use_scaling=True,
    )

    oos_start = train_win_periods
    y_oos = y_chunk[oos_start:]
    dates_oos = dates_chunk.iloc[oos_start:].values
    baselines_oos = baselines_chunk[oos_start:]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results.to_csv(output_file, index=False)
    from src.evaluation import save_chunk_reduce

    save_chunk_reduce(results, output_file)
    print(f"Saved {len(results)} rows -> {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge walk-forward backtest")
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=500, help="training window in days")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--exog-cols", default=None, help="Pipe-separated exog columns, e.g. vix|sentiment")
    parser.add_argument("--segment", default=None, choices=SEGMENT_CHOICES, help="Time-of-day segment")
    parser.add_argument(
        "--lag-scope", default="global", choices=["global", "intra"], help="Compute lags on full dataset or per-segment"
    )
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    exog_cols = parse_exog_cols(args.exog_cols)

    # --- Load and transform ---
    df = load_raw_data(args.data_path, allow_missing=True)
    if exog_cols:
        apply_overnight_fills(df, exog_cols)
        df = df.dropna(subset=["RV"] + exog_cols).reset_index(drop=True)

    adj_rv, baseline = robust_transform(df, "RV", is_target=True)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    adj_exog_cols: list[str] = []
    for col in exog_cols:
        adj_col = f"adj_{col}"
        adj_series, _ = robust_transform(df, col, use_transform=True, use_diurnal=True)
        df[adj_col] = adj_series
        adj_exog_cols.append(adj_col)

    # --- No segment: global backtest ---
    if args.segment is None:
        train_win_periods = args.train_window * PERIODS_PER_DAY
        df, feature_names = generate_har_features(df, target_col="adj_RV", exog_cols=adj_exog_cols)
        _run_backtest_and_save(
            df, feature_names, train_win_periods, args.horizon, args.start, args.end, args.output_file
        )
        return

    # --- Segmented backtest ---
    segments = list(SEGMENT_DEFINITIONS) if args.segment == "all" else [args.segment]

    if args.lag_scope == "global":
        df, feature_names = generate_har_features(df, target_col="adj_RV", exog_cols=adj_exog_cols)

    for seg_name in segments:
        seg_df = slice_to_segment(df, seg_name)
        if seg_df.empty:
            print(f"No data for segment '{seg_name}'. Skipping.")
            continue

        if args.lag_scope == "intra":
            seg_df, feature_names = generate_har_features(seg_df, target_col="adj_RV", exog_cols=adj_exog_cols)

        train_win_periods = compute_segment_train_window(seg_df["t"], args.train_window)

        base, ext = os.path.splitext(args.output_file)
        seg_output = f"{base}_{seg_name}{ext}"

        print(f"{'=' * 20} SEGMENT: {seg_name.upper()} {'=' * 20}")
        print(f"Window: {train_win_periods} periods ({args.train_window} days)")
        _run_backtest_and_save(seg_df, feature_names, train_win_periods, args.horizon, args.start, args.end, seg_output)


if __name__ == "__main__":
    main()
