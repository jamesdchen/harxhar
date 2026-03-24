"""Lag feature generation: resolve lag indices, build features, clean burn-in."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.core import config
from core.features.transforms import HARFeatures, RawLagFeatures

# ---------------------------------------------------------------------------
# Lag resolution
# ---------------------------------------------------------------------------


def resolve_lags(feature_type: str, lag: int) -> list[int]:
    """Return the lag index list for the given feature type and max lag.

    HAR  → geometric base-5 sequence [1, 5, 25, …] up to *lag*.
    pca  → log-spaced lags from 1 to *lag*, dense at short horizons.
    raw  → consecutive lags [1, 2, …, lag].
    """
    if feature_type == "har":
        seq, v = [], 1
        while v <= lag:
            seq.append(v)
            v *= 5
        return seq
    if feature_type == "pca":
        # Log-spaced lags covering same range as HAR but with enough
        # density for PCA to capture the autocorrelation structure.
        import numpy as np

        n_points = 20
        raw = np.geomspace(1, lag, num=n_points)
        seq = sorted(set(int(round(v)) for v in raw))
        return [v for v in seq if 1 <= v <= lag]
    return list(range(1, lag + 1))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_generator(feature_type: str, lags_list: list[int], target_col: str):
    """Instantiate the appropriate feature generator."""
    FeatureClass = HARFeatures if feature_type == "har" else RawLagFeatures
    return FeatureClass(lags=lags_list, target_col=target_col)


def _generate_and_concat(generator, df: pd.DataFrame, cols_to_transform: list[str]):
    """Generate features and concatenate them onto *df*."""
    feat_dict, feature_names = generator.generate_pandas(df, cols_to_transform)
    feat_df = pd.DataFrame(feat_dict, index=df.index)
    return pd.concat([df, feat_df], axis=1), feature_names


def _clean_nans(df: pd.DataFrame, target_col: str, allow_missing: bool, max_lag: int):
    """Burn-in / NaN cleanup after lag feature generation."""
    if allow_missing:
        df = df.iloc[max_lag:]
        df = df.dropna(subset=[target_col, "baseline_RV"]).reset_index(drop=True)
    else:
        df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_lag_features(
    data: pd.DataFrame,
    cols_to_transform: list[str],
    hparams: dict,
    target_col: str = "adj_RV",
    lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray, list[str]]:
    """Build lag features on a pre-loaded DataFrame (global / non-segmented).

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned data with target, baseline, and exogenous columns.
    cols_to_transform : list[str]
        Columns to generate lag features for.
    hparams : dict
        Pipeline hyperparameters (feature_type, is_tree, allow_missing).
    target_col : str
        Name of the target column.
    lag : int or None
        Maximum lag value. Defaults to ``config.LAG``.

    Returns
    -------
    (X, y, dates, baselines, feature_names)
    """
    if lag is None:
        lag = config.LAG

    allow_missing = hparams.get("allow_missing", False)
    feature_type = hparams.get("feature_type", "raw")
    lags_list = resolve_lags(feature_type, lag)

    generator = _make_generator(feature_type, lags_list, target_col)
    data, final_features = _generate_and_concat(generator, data, cols_to_transform)

    if hparams.get("is_tree", False):
        final_features.extend(["DOW", "hour"])

    required_cols = ["t", target_col, "baseline_RV"] + final_features
    data = data[required_cols]
    data = _clean_nans(data, target_col, allow_missing, max(lags_list))

    X_np = data[final_features].values.astype(np.float64)
    y_np = data[target_col].values.astype(np.float64)

    return X_np, y_np, data["t"], data["baseline_RV"].values, final_features


def generate_lag_features_segmented(
    data: pd.DataFrame,
    cols_to_transform: list[str],
    hparams: dict,
    target_segment: str,
    target_col: str = "adj_RV",
    lag: int | None = None,
) -> dict | tuple:
    """Build lag features per segment on a pre-loaded DataFrame.

    Lag calculation strategy is controlled by ``hparams['lag_scope']``:
      - ``'global'`` (default): calculates all lags on the full dataset before
        slicing into segments.
      - ``'intra'``: calculates lags within each segment independently.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned data with target, baseline, and exogenous columns.
    cols_to_transform : list[str]
        Columns to generate lag features for.
    hparams : dict
        Pipeline hyperparameters (feature_type, lag_scope, allow_missing).
    target_segment : str
        ``'all'`` → return dict of all segments.
        segment name → return ``(X, y, dates, baselines)`` for one segment.
    target_col : str
        Name of the target column.
    lag : int or None
        Maximum lag value. Defaults to ``config.LAG``.

    Returns
    -------
    dict or tuple
    """
    if lag is None:
        lag = config.LAG

    allow_missing = hparams.get("allow_missing", False)
    feature_type = hparams.get("feature_type", "raw")
    lags_list = resolve_lags(feature_type, lag)
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

    if target_segment == "all":
        return datasets
    elif target_segment in datasets:
        ds = datasets[target_segment]
        return ds["X"], ds["y"], ds["dates"], ds["baselines"]
    else:
        return np.array([]), np.array([]), [], []
