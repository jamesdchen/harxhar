import numpy as np
import pandas as pd
from src import config
from src.data_main import load_and_clean_base_data
from src.features import HARFeatures, RawLagFeatures

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

    FeatureClass = HARFeatures if feature_type == 'har' else RawLagFeatures
    generator = FeatureClass(lags=config.HAR_LAGS, target_col=target_col)
    new_features_dict, final_features = generator.generate(data, cols_to_transform)

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