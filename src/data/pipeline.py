"""Feature generation pipeline: horizon shifts, lag computation, segmented loading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core import config
from src.core.config import check_positive
from src.data.loading import load_and_clean_base_data
from src.features import HARFeatures, RawLagFeatures

# ---------------------------------------------------------------------------
# Horizon shift utility
# ---------------------------------------------------------------------------


def apply_horizon_shift(X, y, dates, baselines, horizon):
    """
    Shift targets forward by (horizon-1) for direct h-step forecasting.

    At each index t, features X[t] remain unchanged while y[t] becomes the
    value (horizon-1) steps into the future.  Baselines are aligned with the
    target time (needed for Duan smearing); dates are kept at prediction time.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    y : np.ndarray, shape (N,)
    dates : pd.Series of length N
    baselines : np.ndarray of length N
    horizon : int >= 1

    Returns
    -------
    (X, y, dates, baselines) with aligned lengths (N - horizon + 1).
    """
    check_positive(horizon, "horizon")
    if horizon > config.PERIODS_PER_DAY:
        raise ValueError(f"horizon must be <= {config.PERIODS_PER_DAY}, got {horizon}")
    if horizon <= 1:
        return X, y, dates, baselines
    shift = horizon - 1
    return (
        X[:-shift],
        y[shift:],
        dates.iloc[:-shift].reset_index(drop=True),
        baselines[shift:],
    )


# ---------------------------------------------------------------------------
# Feature generation helpers
# ---------------------------------------------------------------------------


def resolve_lags(feature_type, lag):
    """Return the lag index list for the given feature type and max lag."""
    if feature_type == "har":
        # Geometric sequence base 5: [1, 5, 25, …] up to lag
        seq, v = [], 1
        while v <= lag:
            seq.append(v)
            v *= 5
        return seq
    # raw: consecutive lags 1..lag
    return list(range(1, lag + 1))


def _make_generator(feature_type, lags_list, target_col):
    """Instantiate the appropriate feature generator."""
    FeatureClass = HARFeatures if feature_type == "har" else RawLagFeatures
    return FeatureClass(lags=lags_list, target_col=target_col)


def _generate_and_concat(generator, df, cols_to_transform):
    """Generate features and concatenate them onto *df*."""
    feat_dict, feature_names = generator.generate_pandas(df, cols_to_transform)
    feat_df = pd.DataFrame(feat_dict, index=df.index)
    return pd.concat([df, feat_df], axis=1), feature_names


def _clean_nans(df, target_col, allow_missing, max_lag):
    """Burn-in / NaN cleanup shared by global and segmented modes."""
    if allow_missing:
        df = df.iloc[max_lag:]
        df = df.dropna(subset=[target_col, "baseline_RV"]).reset_index(drop=True)
    else:
        df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_prep_data_strided(
    hparams: dict,
    input_path: str,
    target_segment: str | None = None,
    lag: int | None = None,
) -> tuple | dict:
    """
    Generates lag features for backtesting.

    feature_type='har'  (hparams): rolling-mean HAR aggregates (original HARXHAR)
    feature_type='raw'  (default): individual point lags via .shift(lag)

    Parameters
    ----------
    hparams : dict
        Pipeline hyperparameters (feature_type, is_tree, allow_missing, lag_scope).
    input_path : str
        Path to parquet file(s).
    target_segment : str or None
        None          → global mode: returns (X, y, dates, baselines, feature_names).
        'all'         → segmented mode: returns dict of all segments.
        segment name  → segmented mode: returns (X, y, dates, baselines) for one segment.
    lag : int or None
        Maximum lag value. Defaults to config.LAG.
        For HAR: generates a geometric base-5 sequence [1, 5, 25, …] up to lag.
        For raw: generates consecutive lags list(range(1, lag + 1)).
    """
    if lag is None:
        lag = config.LAG

    data, cols_to_transform = load_and_clean_base_data(hparams, input_path)
    if data.empty:
        if target_segment == "all":
            return {}
        return np.array([]), np.array([]), [], []

    target_col = "adj_RV"
    allow_missing = hparams.get("allow_missing", False)
    feature_type = hparams.get("feature_type", "raw")
    lags_list = resolve_lags(feature_type, lag)

    # --- Segmented mode ---
    if target_segment is not None:
        return _load_segmented(
            data,
            cols_to_transform,
            hparams,
            target_segment,
            target_col,
            allow_missing,
            feature_type,
            lags_list,
        )

    # --- Global mode ---
    generator = _make_generator(feature_type, lags_list, target_col)
    data, final_features = _generate_and_concat(generator, data, cols_to_transform)

    # Keep DOW and hour for tree models
    if hparams.get("is_tree", False):
        final_features.extend(["DOW", "hour"])

    # Final Clean & Matrix Extraction
    required_cols = ["t", target_col, "baseline_RV"] + final_features
    data = data[required_cols]
    data = _clean_nans(data, target_col, allow_missing, max(lags_list))

    X_np = data[final_features].values.astype(np.float64)
    y_np = data[target_col].values.astype(np.float64)

    return X_np, y_np, data["t"], data["baseline_RV"].values, final_features


def _load_segmented(
    data, cols_to_transform, hparams, target_segment, target_col, allow_missing, feature_type, lags_list
):
    """
    Internal helper for segmented lag feature generation.

    Lag calculation strategy is controlled by hparams['lag_scope']:
      - 'global'  (default): Calculates all lags on the full dataset before
                             slicing into segments.
      - 'intra'            : Calculates lags within each segment independently.
    """
    lag_scope = hparams.get("lag_scope", "global")
    max_lag = max(lags_list)

    # GLOBAL MODE: pre-compute lags on full dataset before segmenting
    all_feature_names = None
    if lag_scope == "global":
        generator = _make_generator(feature_type, lags_list, target_col)
        feat_dict, all_feature_names = generator.generate_pandas(data, cols_to_transform)
        for name, series in feat_dict.items():
            data[name] = series

    minutes = data["t"].dt.hour * 60 + data["t"].dt.minute
    datasets = {}

    for seg_name, times in config.SEGMENT_DEFINITIONS.items():
        if target_segment not in ["all", seg_name]:
            continue

        start, end = times["start"], times["end"]

        if start < end:
            mask = (minutes >= start) & (minutes <= end)
        else:
            mask = (minutes >= start) | (minutes <= end)

        seg_df = data.loc[mask].copy()
        if seg_df.empty:
            continue

        # INTRA MODE: compute lags per-segment
        if lag_scope == "intra":
            generator = _make_generator(feature_type, lags_list, target_col)
            seg_df, feature_names = _generate_and_concat(generator, seg_df, cols_to_transform)
        else:
            feature_names = all_feature_names

        required_cols = ["t", target_col, "baseline_RV"] + feature_names
        seg_df = seg_df[required_cols]

        # Burn-in / NaN cleanup
        seg_df = _clean_nans(seg_df, target_col, allow_missing, max_lag)

        if seg_df.empty:
            continue

        datasets[seg_name] = {
            "X": seg_df[feature_names].values.astype(np.float64),
            "y": seg_df[target_col].values.astype(np.float64),
            "dates": seg_df["t"],
            "baselines": seg_df["baseline_RV"].values,
            "features": feature_names,
        }

    # Routing return
    if target_segment == "all":
        return datasets
    elif target_segment in datasets:
        ds = datasets[target_segment]
        return ds["X"], ds["y"], ds["dates"], ds["baselines"]
    else:
        return np.array([]), np.array([]), [], []
