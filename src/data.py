import numpy as np
import pandas as pd
from src import config
from src.data_helper import load_and_clean_base_data

def load_and_prep_data_strided(hparams, input_path):
    """
    Generates continuous, global HAR lag features.
    """
    data, cols_to_transform = load_and_clean_base_data(hparams, input_path)
    if data.empty:
        return np.array([]), np.array([]), [], []

    final_features = []
    new_features_dict = {}
    
    # 1. Calculate and store in a dictionary (Fast)
    for col in cols_to_transform:
        for lag in config.HAR_LAGS: # Use HAR_LAGS if not using config.py
            # Preserve original target HAR naming for RV vs Exogenous
            feat_name = f"har_ma_{lag}" if col == 'adj_log_RV' else f"{col}_ma_{lag}"
            
            new_features_dict[feat_name] = data[col].rolling(window=lag).mean().shift(1)
            final_features.append(feat_name)
            
    # 2. Convert dictionary to DataFrame and concatenate all at once (Zero fragmentation)
    new_features_df = pd.DataFrame(new_features_dict, index=data.index)
    data = pd.concat([data, new_features_df], axis=1)

    # --- 3. Final Clean & Matrix Extraction (RESTORED LOGIC) ---
    required_cols = ['t', 'adj_log_RV', 'baseline_RV'] + final_features
    data = data[required_cols]
    
    # Drop rows with NaNs introduced by the longest HAR lags (the burn-in period)
    data = data.dropna().reset_index(drop=True)  
    
    # Extract matrices
    X_np = data[final_features].values.astype(np.float64)
    y_np = data['adj_log_RV'].values.astype(np.float64)
    
    return X_np, y_np, data['t'], data['baseline_RV'].values