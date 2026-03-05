import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
DIURNAL_WINDOW = 20     # 20 Days
DIURNAL_MIN_PERIODS = 5 
# HAR Lags (Geometric Sequence)
HAR_LAGS = [1, 5, 25, 125, 625, 3125] 

def robust_log_diurnal_transform(df, col_name, time_col="time_of_day"):
    """
    Applies 'Log -> Clip -> Rolling Diurnal Adj'.
    Returns (Adjusted_Series, Baseline_Series).
    """
    # 1. Log Transform (Handle zeros)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_vals = np.log(df[col_name])
    
    # 2. Clip Logic (Floor)
    finite_vals = log_vals[np.isfinite(log_vals)]
    if len(finite_vals) > 0:
        floor = finite_vals.min() - 2.0
    else:
        floor = -20.0
        
    log_clipped = log_vals.copy()
    log_clipped[~np.isfinite(log_clipped)] = floor
    
    # 3. Rolling Diurnal Adjustment (Median + Shift)
    baseline = df.groupby(time_col)[col_name].transform(
        lambda x: log_clipped.loc[x.index].rolling(
            window=DIURNAL_WINDOW, min_periods=DIURNAL_MIN_PERIODS
        ).median().shift(1)
    )
    
    # Fill missing baselines (start of dataset)
    baseline = baseline.fillna(method='bfill').fillna(0.0)
    
    # 4. Subtract Baseline
    adj_series = log_clipped - baseline
    
    return adj_series, baseline

def load_and_prep_data_strided(hparams, input_file):
    print(f"Loading {input_file}...")
    try: 
        data = pd.read_parquet(input_file, engine="pyarrow")
    except: 
        data = pd.read_csv(input_file)
        
    # 1. Standardize Time
    if 'endbartime' in data.columns: 
        data = data.rename(columns={'endbartime': 't', 'sumret2': 'RV'})
    
    data['t'] = pd.to_datetime(data['t'])

    # --- CRITICAL FIX: REINDEX TO FULL GRID ---
    # This ensures 1 day = 48 periods, matching your harx_coef.py math.
    print("Aligning to full market grid (48 periods/day)...")
    dates = data['t'].dt.date.unique()
    all_slots = []
    for d in dates:
        # Generate full 24h grid (48 slots of 30min)
        day_slots = pd.date_range(start=f"{d} 00:00", end=f"{d} 23:30", freq="30min")
        all_slots.append(day_slots)
        
    full_grid = pd.DatetimeIndex(np.concatenate(all_slots)).sort_values()
    
    # Reindex forces creation of night rows (filled with NaN initially)
    data = data.set_index('t').reindex(full_grid)
    data.index.name = 't'
    data = data.reset_index()

    # --- GAP FILLING ---
    # 1. Fill Target: Nighttime Volatility is effectively 0
    data['RV'] = data['RV'].fillna(0.0)
    
    # 2. Identify Exogenous Columns EARLY
    exog_cols = []
    exog_feat_names = []
    
    if hparams.get("exog_cols") and str(hparams["exog_cols"]).lower() != "none":
        sep = '|' if '|' in hparams["exog_cols"] else ','
        raw_exog_list = hparams["exog_cols"].split(sep)
        
        # Verify columns exist
        exog_cols = [c.strip() for c in raw_exog_list if c.strip() in data.columns]
        
        if exog_cols:
            print(f"Filling gaps for exogenous cols: {exog_cols}")
            # 3. Fill Exog: Forward Fill (carry close price/level overnight)
            data[exog_cols] = data[exog_cols].fillna(method='ffill').fillna(0.0)

    data['time_of_day'] = data['t'].dt.time
    
    # 2. Process Target (RV) -> adj_log_RV
    print("Applying Robust Diurnal Adj to Target (RV)...")
    data['adj_log_RV'], data['baseline_RV'] = robust_log_diurnal_transform(data, 'RV', 'time_of_day')
    
    # 3. Process Exogenous Features (Generate Lags)
    for raw_col in exog_cols:
        print(f"Generating HAR Lags for {raw_col}...")
        
        # A. Transform the raw series
        base_adj_col = f"adj_log_{raw_col}"
        data[base_adj_col], _ = robust_log_diurnal_transform(data, raw_col, 'time_of_day')
        
        # B. Generate HAR Lags for this Exogenous Feature
        for lag in HAR_LAGS:
            feat_name = f"{base_adj_col}_ma_{lag}"
            
            # Calculate Rolling Mean of the Adjusted Series
            data[feat_name] = data[base_adj_col].rolling(window=lag).mean().shift(1)
            exog_feat_names.append(feat_name)

    # 4. Create HAR Features for Target (RV)
    har_features = []
    for lag in HAR_LAGS:
        feat_name = f"har_ma_{lag}"
        data[feat_name] = data['adj_log_RV'].rolling(window=lag).mean().shift(1)
        har_features.append(feat_name)
    
    pd.set_option('display.max_columns', None)
    start_cutoff = pd.Timestamp("2003-01-04 00:00:00")
    print(data[data['t'] >= start_cutoff])
    
    # 5. Clean & Finalize
    # Drop rows that have NaNs due to the longest rolling window
    # Now that grid is continuous, this correctly drops just the first ~65 days (max lag)
    
    final_cols = har_features + exog_feat_names
    print(f"Final Features ({len(final_cols)}): {final_cols}")
    
    required_cols = ['t', 'adj_log_RV', 'baseline_RV'] + final_cols
    
    # Filter data to remove raw exogenous columns and intermediate 'RV' columns
    data = data[required_cols]

    # Now drop rows that have NaNs (the rolling window lead-in period)
    data = data.dropna()
    
    print(f"Post-cleaning shape: {data.shape}")
    print(data[data['t'] >= start_cutoff].head())
    
    # NEEDED FOR CHUNK ID ACCURACY AFTER SCRUBBING
    data = data.reset_index(drop=True)
    
    # Construct Feature Matrix X
    
    X_np = data[final_cols].values.astype(np.float64)
    y_np = data['adj_log_RV'].values.astype(np.float64)
    
    # Auxiliary data for reconstruction
    # Keep dates as index or series aligned with X_np
    dates = data['t'].values
    baselines = data['baseline_RV'].values
    
    # Return updated signature
    return X_np, y_np, dates, baselines, final_cols

def get_chunk_indices_strided(X_np, train_window_size, chunk_id, total_chunks):
    """
    Returns indices for the TEST portion of the pooled rolling window.
    """
    num_samples = X_np.shape[0]
    valid_test_start = train_window_size
    
    if valid_test_start >= num_samples:
        return np.array([])
    
    test_indices = np.arange(valid_test_start, num_samples)
    chunk_indices_list = np.array_split(test_indices, total_chunks)
    
    if chunk_id >= len(chunk_indices_list):
        return np.array([])
        
    return chunk_indices_list[chunk_id]

def save_chunk_results(output_file, forecasts, naive, indices, train_window, y_true, dates, baselines):
    """
    Saves results including reconstructed RAW predictions with Log-Normal Bias Correction.
    """
    y_subset = y_true[indices]
    
    # Handle dates regardless of if they are Index, Series, or Array
    if hasattr(dates, 'iloc'):
        dates_subset = dates.iloc[indices].values
    else:
        dates_subset = dates[indices]
        
    base_subset = baselines[indices]
    
    # Log-Normal Bias Correction (sigma^2 / 2)
    sigma2 = np.var(y_subset - forecasts)
    pred_raw = np.exp(forecasts + base_subset + (sigma2 / 2))
    true_raw = np.exp(y_subset + base_subset)
    
    df = pd.DataFrame({
        'date': dates_subset,
        'true_adj': y_subset,
        'pred_adj': forecasts,
        'true_raw': true_raw,
        'pred_raw': pred_raw
    })
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return dates_subset