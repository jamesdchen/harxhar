"""Ridge regression volatility backtest executor.

Self-contained walk-forward backtest with rolling robust scaling,
HAR lag features, and Duan smearing.  No imports from core/ or projects/.
"""

import argparse
import os

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import Ridge
from tqdm import tqdm

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


def generate_har_features(
    df: pd.DataFrame, target_col: str = "adj_RV"
) -> tuple[pd.DataFrame, list[str]]:
    lags = resolve_har_lags()
    features: dict[str, pd.Series] = {}
    feature_names: list[str] = []
    for lag in lags:
        name = f"har_ma_{lag}"
        features[name] = (
            df[target_col].rolling(window=lag, min_periods=1).mean().shift(1)
        )
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


# ── Numba kernels for rolling robust scaling ──────────────────────────────

@njit(cache=True)
def _update_sorted_matrix(
    sorted_mat: np.ndarray, x_old: np.ndarray, x_new: np.ndarray
) -> None:
    n_features, w = sorted_mat.shape
    for i in range(n_features):
        v_old = x_old[i]
        v_new = x_new[i]
        idx_old = np.searchsorted(sorted_mat[i], v_old)
        idx_new = np.searchsorted(sorted_mat[i], v_new)
        if idx_old < idx_new:
            idx_new -= 1
            for j in range(idx_old, idx_new):
                sorted_mat[i, j] = sorted_mat[i, j + 1]
        elif idx_old > idx_new:
            for j in range(idx_old, idx_new, -1):
                sorted_mat[i, j] = sorted_mat[i, j - 1]
        sorted_mat[i, idx_new] = v_new


@njit(cache=True)
def _get_robust_stats(
    sorted_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_features, w = sorted_mat.shape
    median = np.empty(n_features, dtype=np.float64)
    iqr = np.empty(n_features, dtype=np.float64)
    idx_25 = (w - 1) * 0.25
    idx_50 = (w - 1) * 0.50
    idx_75 = (w - 1) * 0.75
    i25_floor, rem_25 = int(idx_25), idx_25 - int(idx_25)
    i50_floor, rem_50 = int(idx_50), idx_50 - int(idx_50)
    i75_floor, rem_75 = int(idx_75), idx_75 - int(idx_75)
    for i in range(n_features):
        q25 = (
            sorted_mat[i, i25_floor] * (1.0 - rem_25)
            + sorted_mat[i, min(i25_floor + 1, w - 1)] * rem_25
        )
        med = (
            sorted_mat[i, i50_floor] * (1.0 - rem_50)
            + sorted_mat[i, min(i50_floor + 1, w - 1)] * rem_50
        )
        q75 = (
            sorted_mat[i, i75_floor] * (1.0 - rem_75)
            + sorted_mat[i, min(i75_floor + 1, w - 1)] * rem_75
        )
        median[i] = med
        iq = q75 - q25
        iqr[i] = iq if iq >= 1e-12 else 1.0
    return median, iqr


# ── RollingRobustScaler ──────────────────────────────────────────────────

class RollingRobustScaler:
    """Maintains a sorted buffer per feature for O(W) median/IQR scaling."""

    def __init__(self, window_size: int, n_features: int) -> None:
        self.window_size = window_size
        self.n_features = n_features
        self.chrono_buf = np.zeros((window_size, n_features), dtype=np.float64)
        self.sorted_mat = np.zeros((n_features, window_size), dtype=np.float64)
        self.pos = 0

    def initialize(self, data_block: np.ndarray) -> None:
        """Fill buffers from *data_block* (shape [window_size, n_features])."""
        w = self.window_size
        self.chrono_buf[:] = data_block[:w]
        for i in range(self.n_features):
            self.sorted_mat[i] = np.sort(data_block[:w, i])
        self.pos = 0

    def update(self, x_new: np.ndarray) -> None:
        """Slide window: replace oldest row with *x_new*."""
        x_old = self.chrono_buf[self.pos].copy()
        self.chrono_buf[self.pos] = x_new
        _update_sorted_matrix(self.sorted_mat, x_old, x_new)
        self.pos = (self.pos + 1) % self.window_size

    def get_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (median, iqr) arrays from the current sorted buffer."""
        return _get_robust_stats(self.sorted_mat)


# ── RollingBuffer ─────────────────────────────────────────────────────────

class RollingBuffer:
    """Ring buffer for (X, y) pairs."""

    def __init__(
        self, window_size: int, n_features: int, n_targets: int = 1
    ) -> None:
        self.window_size = window_size
        self.X = np.zeros((window_size, n_features), dtype=np.float64)
        self.y = np.zeros((window_size, n_targets), dtype=np.float64)
        self.pos = 0
        self.count = 0

    def add(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        self.X[self.pos] = x_new
        self.y[self.pos] = y_new
        self.pos = (self.pos + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)

    def get_view(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < self.window_size:
            return self.X[: self.count], self.y[: self.count]
        idx = np.roll(np.arange(self.window_size), -self.pos)
        return self.X[idx], self.y[idx]


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
    refit_frequency: int = 1,
    use_scaling: bool = True,
) -> np.ndarray:
    """Walk-forward backtest returning an array of predictions.

    Parameters
    ----------
    model_fn : callable
        Returns a *new* sklearn estimator (unfitted).
    X, y : np.ndarray
        Full feature / target arrays.
    train_win : int
        Number of initial rows used for the first fit.
    refit_frequency : int
        Re-estimate the model every *refit_frequency* steps.
    use_scaling : bool
        If True, apply rolling robust scaling to X.
    """
    n_samples, n_features = X.shape
    predictions = np.full(n_samples - train_win, np.nan)

    # 1. Initialise scaler + buffer
    scaler = RollingRobustScaler(train_win, n_features) if use_scaling else None
    buf = RollingBuffer(train_win, n_features, n_targets=1)

    X_init = X[:train_win].copy()
    y_init = y[:train_win].copy()

    if use_scaling:
        scaler.initialize(X_init)
        med, iqr = scaler.get_scaler()
        X_scaled_init = (X_init - med) / iqr
    else:
        X_scaled_init = X_init

    for i in range(train_win):
        buf.add(X_scaled_init[i], y_init[i : i + 1])

    # 2. Fit initial model
    X_buf, y_buf = buf.get_view()
    model = model_fn()
    model.fit(X_buf, y_buf.ravel())

    # 3. Walk forward
    for t in tqdm(range(train_win, n_samples), desc="backtest"):
        x_t_raw = X[t]

        # a. Scale
        if use_scaling:
            med, iqr = scaler.get_scaler()
            x_t_scaled = (x_t_raw - med) / iqr
        else:
            x_t_scaled = x_t_raw

        # b. Predict
        predictions[t - train_win] = model.predict(x_t_scaled.reshape(1, -1))[0]

        # c. Update scaler with raw observation
        if use_scaling:
            scaler.update(x_t_raw)

        # d. Add scaled observation to buffer
        buf.add(x_t_scaled, y[t : t + 1])

        # e. Refit
        if (t - train_win + 1) % refit_frequency == 0:
            X_buf, y_buf = buf.get_view()
            model = model_fn()
            model.fit(X_buf, y_buf.ravel())

    return predictions


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge walk-forward backtest")
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--train-window", type=int, default=500, help="training window in days"
    )
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--total-chunks", type=int, default=1)
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

    # 7. Chunk split
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
        raise ValueError(
            f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})"
        )

    # 8. Walk-forward backtest
    model_fn = lambda: Ridge(alpha=1.0)
    preds = run_backtest(
        model_fn,
        X_chunk,
        y_chunk,
        train_win=train_win_periods,
        refit_frequency=1,
        use_scaling=True,
    )

    # 9. Duan smearing + save
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
    print(f"Saved {len(results)} rows → {args.output_file}")


if __name__ == "__main__":
    main()
