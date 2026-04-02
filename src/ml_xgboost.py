"""XGBoost volatility backtest executor.

Self-contained walk-forward backtest with HAR lag features, DOW/hour
features, and Duan smearing.  No imports from core/ or projects/.
Tree-based model: no scaling, handles NaN natively.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor

from evaluation import calculate_metrics
from src.loading import load_raw_data
from src.transforms import robust_transform

# ── Constants ─────────────────────────────────────────────────────────────
PERIODS_PER_DAY = 48


# ── HAR lag features ─────────────────────────────────────────────────────


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


# ── DOW + hour features ─────────────────────────────────────────────────


def add_calendar_features(df: pd.DataFrame) -> list[str]:
    """Add day-of-week (0-6) and hour features. Returns new column names."""
    df["DOW"] = df["t"].dt.dayofweek
    df["hour"] = df["t"].dt.hour
    return ["DOW", "hour"]


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


# ── RollingBuffer ─────────────────────────────────────────────────────────


class RollingBuffer:
    """Ring buffer for (X, y) pairs."""

    def __init__(self, window_size: int, n_features: int) -> None:
        self.window_size = window_size
        self.ptr = 0
        self.count = 0
        self.X_buffer = np.zeros((window_size, n_features))
        self.y_buffer = np.zeros((window_size, 1))

    def add(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        self.X_buffer[self.ptr] = x_new
        self.y_buffer[self.ptr] = y_new
        self.ptr = (self.ptr + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1

    def get_view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X_buffer, self.y_buffer


# ── Duan smearing (inline) ───────────────────────────────────────────────


def apply_duan_smearing(
    forecasts: np.ndarray, y_true: np.ndarray, baselines: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    smear = np.mean((y_true - forecasts) ** 2)
    pred_raw = (forecasts**2 + smear) * baselines
    true_raw = (y_true**2) * baselines
    return pred_raw, true_raw


# ── Walk-forward backtest ─────────────────────────────────────────────────


def run_backtest(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    refit_frequency: int = 5,
) -> np.ndarray:
    """Walk-forward backtest returning an array of predictions.

    Parameters
    ----------
    model_fn : callable
        Returns a *new* XGBRegressor (unfitted).
    X, y : np.ndarray
        Full feature / target arrays.
    train_win : int
        Number of initial rows used for the first fit.
    refit_frequency : int
        Re-estimate the model every *refit_frequency* steps.
    """
    n_samples, n_features = X.shape
    predictions = np.full(n_samples - train_win, np.nan)

    # 1. Initialise buffer
    buf = RollingBuffer(train_win, n_features)

    for i in range(train_win):
        buf.add(X[i], y[i : i + 1])

    # 2. Fit initial model
    X_buf, y_buf = buf.get_view()
    model = model_fn()
    model.fit(X_buf, y_buf.ravel())

    # 3. Walk forward
    for t in tqdm(range(train_win, n_samples), desc="backtest"):
        x_t = X[t]

        # a. Predict (no scaling)
        predictions[t - train_win] = model.predict(x_t.reshape(1, -1))[0]

        # b. Add observation to buffer
        buf.add(x_t, y[t : t + 1])

        # c. Refit every refit_frequency steps
        if (t - train_win + 1) % refit_frequency == 0:
            X_buf, y_buf = buf.get_view()
            model = model_fn()
            model.fit(X_buf, y_buf.ravel())

    return predictions


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost walk-forward backtest")
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=500, help="training window in days")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--total-chunks", type=int, default=1)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    train_win_periods = args.train_window * PERIODS_PER_DAY

    # 1. Load data (allow_missing=True for tree models)
    df = load_raw_data(args.data_path, allow_missing=True)

    # 2. Robust transform on RV (full transform for target)
    adj_rv, baseline = robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=240)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    # 3. HAR features (no transform/diurnal on exog)
    df, har_names = generate_har_features(df, target_col="adj_RV")

    # 4. DOW + hour features
    cal_names = add_calendar_features(df)

    feature_names = har_names + cal_names

    # 5. Drop initial NaN rows from HAR lag computation
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    # 6. Extract numpy arrays
    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    # 7. Horizon shift
    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, args.horizon)

    # 8. Chunk split
    n = len(X)
    chunk_size = n // args.total_chunks
    start = args.chunk_id * chunk_size
    end = n if args.chunk_id == args.total_chunks - 1 else start + chunk_size

    X_chunk = X[start:end]
    y_chunk = y[start:end]
    dates_chunk = dates.iloc[start:end].reset_index(drop=True)
    baselines_chunk = baselines[start:end]

    # Ensure train window fits in chunk
    if train_win_periods >= len(X_chunk):
        raise ValueError(f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})")

    # 9. Walk-forward backtest
    def model_fn() -> XGBRegressor:
        return XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
        )

    preds = run_backtest(
        model_fn,
        X_chunk,
        y_chunk,
        train_win=train_win_periods,
        refit_frequency=5,
    )

    # 10. Duan smearing + save
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
    metrics_path = os.path.join(out_dir, f"metrics_chunk_{args.chunk_id + 1}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Saved {len(results)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
