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

    # --- NEW: Check toggle and set target column name dynamically ---
    use_log = hparams.get('use_log', True)
    target_col = 'adj_log_RV' if use_log else 'adj_RV'

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
                feat_name = f"har_ma_{lag}" if col == target_col else f"{col}_ma_{lag}"
                
                # Store the series in the dictionary instead of inserting into seg_df
                # In load_and_prep_data_strided (global and TOD versions)
                new_feats_dict[feat_name] = data[col].rolling(
                    window=lag, 
                    min_periods=1 # <-- ADD THIS to prevent NaN cascading
                ).mean().shift(1)
                segment_features.append(feat_name)
        
        # Concatenate all new columns to the segment dataframe at once
        new_feats_df = pd.DataFrame(new_feats_dict, index=seg_df.index)
        seg_df = pd.concat([seg_df, new_feats_df], axis=1)
        
        # --- THE FIX: Retain base columns + all the new features ---
        required_cols = ['t', target_col, 'baseline_RV'] + segment_features
        seg_df = seg_df[required_cols]
        
        allow_missing = hparams.get('allow_missing', False)
    
        if allow_missing:
            # SNIPER: Drop only the burn-in rows and rows with missing targets
            max_lag = max(config.HAR_LAGS)
            seg_df = seg_df.iloc[max_lag:] # Slice off the initial burn-in
            seg_df = seg_df.dropna(subset=[target_col, 'baseline_RV']).reset_index(drop=True)
        else:
            # SHOTGUN: Drop everything (for Ridge)
            seg_df = seg_df.dropna().reset_index(drop=True)
            
        if seg_df.empty:
            return np.array([]), np.array([]), [], []

        datasets[seg_name] = {
            'X': seg_df[segment_features].values.astype(np.float64),
            'y': seg_df[target_col].values.astype(np.float64), # Updated target selection
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