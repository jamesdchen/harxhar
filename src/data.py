import numpy as np
import pandas as pd
from src import config
from src.data_main import load_and_clean_base_data
from src.features import HARFeatures, RawLagFeatures


def load_and_prep_data_strided(
    hparams: dict,
    input_path: str,
    target_segment: str | None = None,
    lags: list[int] | None = None,
) -> tuple | dict:
    """
    Generates lag features for backtesting.

    feature_type='har'  (hparams): rolling-mean HAR aggregates (original HARXHAR)
    feature_type='raw'  (default): individual point lags via .shift(lag)

    Parameters
    ----------
    hparams : dict
        Pipeline hyperparameters (feature_type, use_transform, allow_missing, lag_scope).
    input_path : str
        Path to parquet file(s).
    target_segment : str or None
        None          → global mode: returns (X, y, dates, baselines, feature_names).
        'all'         → segmented mode: returns dict of all segments.
        segment name  → segmented mode: returns (X, y, dates, baselines) for one segment.
    lags : list[int] or None
        Lag indices to use. Defaults to config.HAR_LAGS.
        Pass list(range(1, N+1)) for dense consecutive lags (e.g. for DL models).
    """
    if lags is None:
        lags = config.HAR_LAGS

    data, cols_to_transform = load_and_clean_base_data(hparams, input_path)
    if data.empty:
        if target_segment is None:
            return np.array([]), np.array([]), [], []
        elif target_segment == 'all':
            return {}
        else:
            return np.array([]), np.array([]), [], []

    target_col = 'adj_RV'
    allow_missing = hparams.get('allow_missing', False)
    feature_type = hparams.get('feature_type', 'raw')

    # --- Segmented mode ---
    if target_segment is not None:
        return _load_segmented(
            data, cols_to_transform, hparams, target_segment,
            target_col, allow_missing, feature_type, lags,
        )

    # --- Global mode ---
    FeatureClass = HARFeatures if feature_type == 'har' else RawLagFeatures
    generator = FeatureClass(lags=config.HAR_LAGS, target_col=target_col)
    new_features_dict, final_features = generator.generate(data, cols_to_transform)

    new_features_df = pd.DataFrame(new_features_dict, index=data.index)
    data = pd.concat([data, new_features_df], axis=1)

    # Keep DOW and hour for tree models
    if hparams.get('use_transform', False):
        final_features.extend(['DOW', 'hour'])

    # Final Clean & Matrix Extraction
    required_cols = ['t', target_col, 'baseline_RV'] + final_features
    data = data[required_cols]

    if allow_missing:
        max_lag = max(lags)
        data = data.iloc[max_lag:]
        data = data.dropna(subset=[target_col, 'baseline_RV']).reset_index(drop=True)
    else:
        data = data.dropna().reset_index(drop=True)

    X_np = data[final_features].values.astype(np.float64)
    y_np = data[target_col].values.astype(np.float64)

    return X_np, y_np, data['t'], data['baseline_RV'].values, final_features


def _load_segmented(data, cols_to_transform, hparams, target_segment,
                    target_col, allow_missing, feature_type, lags):
    """
    Internal helper for segmented lag feature generation.

    Lag calculation strategy is controlled by hparams['lag_scope']:
      - 'global'  (default): Calculates all lags on the full dataset before
                             slicing into segments.
      - 'intra'            : Calculates lags within each segment independently.
    """
    lag_scope = hparams.get('lag_scope', 'global')

    # GLOBAL MODE: pre-compute lags on full dataset before segmenting
    all_feature_names = None
    if lag_scope == 'global':
        FeatureClass = HARFeatures if feature_type == 'har' else RawLagFeatures
        generator = FeatureClass(lags=config.HAR_LAGS, target_col=target_col)
        feat_dict, all_feature_names = generator.generate(data, cols_to_transform)
        for name, series in feat_dict.items():
            data[name] = series

    minutes = data['t'].dt.hour * 60 + data['t'].dt.minute
    datasets = {}

    for seg_name, times in config.SEGMENT_DEFINITIONS.items():
        if target_segment not in ['all', seg_name]:
            continue

        start, end = times['start'], times['end']

        if start < end:
            mask = (minutes >= start) & (minutes <= end)
        else:
            mask = (minutes >= start) | (minutes <= end)

        seg_df = data.loc[mask].copy()
        if seg_df.empty:
            continue

        # INTRA MODE: compute lags per-segment
        if lag_scope == 'intra':
            FeatureClass = HARFeatures if feature_type == 'har' else RawLagFeatures
            seg_generator = FeatureClass(lags=config.HAR_LAGS, target_col=target_col)
            new_feats_dict, segment_features = seg_generator.generate(seg_df, cols_to_transform)
            new_feats_df = pd.DataFrame(new_feats_dict, index=seg_df.index)
            seg_df = pd.concat([seg_df, new_feats_df], axis=1)
            feature_names = segment_features
        else:
            feature_names = all_feature_names

        required_cols = ['t', target_col, 'baseline_RV'] + feature_names
        seg_df = seg_df[required_cols]

        # Burn-in / NaN cleanup
        if allow_missing:
            max_lag = max(config.HAR_LAGS)
            seg_df = seg_df.iloc[max_lag:]
            seg_df = seg_df.dropna(subset=[target_col, 'baseline_RV']).reset_index(drop=True)
        else:
            seg_df = seg_df.dropna().reset_index(drop=True)

        if seg_df.empty:
            continue

        datasets[seg_name] = {
            'X': seg_df[feature_names].values.astype(np.float64),
            'y': seg_df[target_col].values.astype(np.float64),
            'dates': seg_df['t'],
            'baselines': seg_df['baseline_RV'].values,
            'features': feature_names
        }

    # Routing return
    if target_segment == 'all':
        return datasets
    elif target_segment in datasets:
        ds = datasets[target_segment]
        return ds['X'], ds['y'], ds['dates'], ds['baselines']
    else:
        return np.array([]), np.array([]), [], []
