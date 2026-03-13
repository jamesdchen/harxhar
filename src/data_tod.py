import numpy as np
import pandas as pd
from src import config
from src.data_main import load_and_clean_base_data

def load_and_prep_data_strided(hparams, input_path, target_segment=None):
    """
    Generates segmented lag features.

    feature_type='har'  (hparams): rolling-mean HAR aggregates (original HARXHAR)
    feature_type='raw'  (default): individual point lags via .shift(lag)

    Lag calculation strategy is controlled by hparams['lag_scope']:
      - 'global'  (default): Calculates all lags on the full dataset before
                             slicing into segments, ensuring temporal continuity
                             across segment boundaries.
      - 'intra'            : Calculates lags within each segment independently,
                             so lags never bleed across segment boundaries.

    If target_segment is 'all', returns a dict of all segments.
    If target_segment is a specific name (e.g., 'morning'), returns (X, y, dates, baselines).
    """
    data, cols_to_transform = load_and_clean_base_data(hparams, input_path)
    if data.empty:
        return {} if target_segment == 'all' else (np.array([]), np.array([]), [], [])

    target_col = 'adj_RV'
    allow_missing = hparams.get('allow_missing', False)
    lag_scope = hparams.get('lag_scope', 'global')
    feature_type = hparams.get('feature_type', 'raw')

    def _make_features(src_df, cols):
        """Return (feat_dict, feat_names) for the given source dataframe."""
        feat_dict, feat_names = {}, []
        for col in cols:
            for lag in config.HAR_LAGS:
                if feature_type == 'har':
                    name = f"har_ma_{lag}" if col == target_col else f"{col}_ma_{lag}"
                    feat_dict[name] = src_df[col].rolling(window=lag, min_periods=1).mean().shift(1)
                else:  # 'raw'
                    name = f"{col}_lag_{lag}"
                    feat_dict[name] = src_df[col].shift(lag)
                feat_names.append(name)
        return feat_dict, feat_names

    # --- GLOBAL MODE: pre-compute lags on full dataset before segmenting ---
    if lag_scope == 'global':
        feat_dict, all_feature_names = _make_features(data, cols_to_transform)
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

        # --- INTRA MODE: compute lags per-segment using the full series for context ---
        if lag_scope == 'intra':
            new_feats_dict, segment_features = _make_features(seg_df, cols_to_transform)
            new_feats_df = pd.DataFrame(new_feats_dict, index=seg_df.index)
            seg_df = pd.concat([seg_df, new_feats_df], axis=1)
            feature_names = segment_features
        else:
            feature_names = all_feature_names

        required_cols = ['t', target_col, 'baseline_RV'] + feature_names
        seg_df = seg_df[required_cols]

        # --- Burn-in / NaN cleanup ---
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

    # --- Routing return ---
    if target_segment == 'all':
        return datasets
    elif target_segment in datasets:
        ds = datasets[target_segment]
        return ds['X'], ds['y'], ds['dates'], ds['baselines']
    else:
        return np.array([]), np.array([]), [], []