import numpy as np
import pandas as pd
from src import config
from src.data_helper import load_and_clean_base_data

def load_and_prep_data_strided(hparams, input_path, target_segment=None):
    """
    Generates segmented HAR lags.
    If target_segment is 'all', returns a dictionary of all segments.
    If target_segment is a specific name (e.g., 'morning'), returns matrices.
    """
    data, cols_to_transform = load_and_clean_base_data(hparams, input_path)
    if data.empty:
        return {} if target_segment == 'all' else (np.array([]), np.array([]), [], [])

    minutes = data['t'].dt.hour * 60 + data['t'].dt.minute
    datasets = {}

    for seg_name, times in config.SEGMENT_DEFINITIONS.items():
        if target_segment not in ['all', seg_name]:
            continue

        start, end = times['start'], times['end']
        
        # Isolate Segment
        if start < end:
            mask = (minutes >= start) & (minutes <= end) 
        else:
            mask = (minutes >= start) | (minutes <= end)
            
        seg_df = data.loc[mask].copy()        
        if seg_df.empty:
            continue
            
        # Generate Intra-Segment Lags
        segment_features = []
        new_feats_dict = {} # Temporary container to prevent fragmentation
        
        for col in cols_to_transform:
            for lag in config.HAR_LAGS:
                # Keep naming consistent with the global script
                feat_name = f"har_ma_{lag}" if col == 'adj_log_RV' else f"{col}_ma_{lag}"
                
                # Store the series in the dictionary instead of inserting into seg_df
                new_feats_dict[feat_name] = seg_df[col].rolling(window=lag, min_periods=1).mean().shift(1)
                segment_features.append(feat_name)
        
        # Concatenate all new columns to the segment dataframe at once
        new_feats_df = pd.DataFrame(new_feats_dict, index=seg_df.index)
        seg_df = pd.concat([seg_df, new_feats_df], axis=1)
        
        # --- THE FIX: Retain base columns + all the new features ---
        required_cols = ['t', 'adj_log_RV', 'baseline_RV'] + segment_features
        seg_df = seg_df[required_cols]
        
        # Clean this specific dataset
        seg_df = seg_df.dropna().reset_index(drop=True)
        if seg_df.empty:
            continue

        datasets[seg_name] = {
            'X': seg_df[segment_features].values.astype(np.float64),
            'y': seg_df['adj_log_RV'].values.astype(np.float64),
            'dates': seg_df['t'],
            'baselines': seg_df['baseline_RV'].values,
            'features': segment_features
        }

    # Routing Return
    if target_segment == 'all':
        return datasets
    elif target_segment in datasets:
        ds = datasets[target_segment]
        return ds['X'], ds['y'], ds['dates'], ds['baselines']
    else:
        raise ValueError(f"Segment '{target_segment}' not found or has no valid data.")