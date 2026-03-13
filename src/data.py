import numpy as np
import pandas as pd
from src import config
from src.data_main import load_and_clean_base_data

def load_and_prep_data_strided(hparams, input_path):
    """
    Generates continuous, global lag features.

    feature_type='har'  (hparams): rolling-mean HAR aggregates (original HARXHAR)
    feature_type='raw'  (default): individual point lags via .shift(lag)
    """
    data, cols_to_transform = load_and_clean_base_data(hparams, input_path)
    if data.empty:
        return np.array([]), np.array([]), [], []

    target_col = 'adj_RV'
    feature_type = hparams.get('feature_type', 'raw')
    
    final_features = []
    new_features_dict = {}

    # 1. Calculate and store in a dictionary (Fast)
    for col in cols_to_transform:
        for lag in config.HAR_LAGS:
            if feature_type == 'har':
                feat_name = f"har_ma_{lag}" if col == target_col else f"{col}_ma_{lag}"
                new_features_dict[feat_name] = data[col].rolling(
                    window=lag,
                    min_periods=1
                ).mean().shift(1)
            else:  # 'raw'
                feat_name = f"{col}_lag_{lag}"
                new_features_dict[feat_name] = data[col].shift(lag)
            final_features.append(feat_name)
            
    # 2. Convert dictionary to DataFrame and concatenate all at once (Zero fragmentation)
    new_features_df = pd.DataFrame(new_features_dict, index=data.index)
    data = pd.concat([data, new_features_df], axis=1)

    # --- NEW: Keep DOW and hour for tree models ---
    if hparams.get('use_transform', False):
        # Since they already exist in 'data', we just add them to our feature list
        final_features.extend(['DOW', 'hour'])

    # --- 3. Final Clean & Matrix Extraction ---
    # Use dynamic target_col here too
    required_cols = ['t', target_col, 'baseline_RV'] + final_features
    data = data[required_cols]
    
    allow_missing = hparams.get('allow_missing', False)
    
    if allow_missing:
        # SNIPER: Drop only the burn-in rows and rows with missing targets
        max_lag = max(config.HAR_LAGS)
        data = data.iloc[max_lag:] # Slice off the initial burn-in
        data = data.dropna(subset=[target_col, 'baseline_RV']).reset_index(drop=True)
    else:
        # SHOTGUN: Drop everything (for Ridge)
        data = data.dropna().reset_index(drop=True)
    
    # Extract matrices using dynamic target_col
    X_np = data[final_features].values.astype(np.float64)
    y_np = data[target_col].values.astype(np.float64)
    
    return X_np, y_np, data['t'], data['baseline_RV'].values, final_features