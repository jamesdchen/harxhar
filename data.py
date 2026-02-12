import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
DIURNAL_WINDOW = 20    # 20 Days
DIURNAL_MIN_PERIODS = 5 
# HAR Lags (Geometric Sequence)
HAR_LAGS = [1, 5, 25, 125, 625, 3125] 

def robust_log_diurnal_transform(df, col_name, time_col="time_of_day", 
                                 diurnal_window=20, min_periods=5):
    """
    Intelligently applies transformations based on feature type.
    
    Logic:
      1. Time Features (hour, DOW) -> PASS-THROUGH (No change)
      2. Signed/Directional (Returns, Sentiment, Net Demand) -> PASS-THROUGH (No Log, No Diurnal)
      3. Magnitude/Activity (Vol, Volume, Spreads, VIX) -> LOG + CLIP + ROBUST DIURNAL ADJ
    
    Returns:
      (transformed_series, baseline_series)
    """
    
    # --- 1. DEFINE CATEGORIES ---
    
    # Features to skip entirely (Metadata/Time)
    SKIP_VARS = {'hour', 'DOW', 't', 'date'}
    
    # Features that can be negative or should not be de-trended (Directional)
    # Note: 'sumret3' is Skewness (Signed), 'sumautocov' is Covariance (Signed)
    SIGNED_KEYWORDS = [
        'sumret',       # Catches sumret, sumret3 (but we must exclude ret2/4/abs below)
        'autocov',      # Covariance
        'sentiment',    # StockTwits Sentiment (-1 to 1)
        'voldemand'     # Net Volatility Demand (Buy - Sell)
    ]
    
    # Exceptions: Variables that contain "sumret" but are actually Positive/Magnitude
    MAGNITUDE_EXCEPTIONS = ['sumret2', 'sumret4', 'sumabsret', 'sumpret2']

    # --- 2. CLASSIFICATION LOGIC ---
    
    # A. Check for Pass-Through
    if col_name in SKIP_VARS:
        return df[col_name], pd.Series(0, index=df.index)

    # B. Check for Signed/Directional
    # It is signed if it matches a keyword AND is not in the exceptions list
    is_signed = False
    for kw in SIGNED_KEYWORDS:
        if kw in col_name:
            # Check if it's actually an exception (e.g. "sumret2" contains "sumret")
            if not any(ex in col_name for ex in MAGNITUDE_EXCEPTIONS):
                is_signed = True
                break
    
    # --- CRITICAL FIX: FORCE NUMERIC ---
    # Convert column to numeric, turning text/errors into NaNs
    series_clean = pd.to_numeric(df[col_name], errors='coerce')

    if is_signed:
        print(f"  [Skipping Diurnal] '{col_name}' identified as Directional/Signed.")
        return series_clean.fillna(0.0), pd.Series(0.0, index=df.index)

    # --- 3. APPLY MAGNITUDE TRANSFORM ---
    
    # A. Log Transform (on the clean numeric series)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_vals = np.log(series_clean)
        
    # B. Clip Logic (Floor)
    # Handle -inf (log(0)) by finding the minimum valid log value and subtracting a buffer
    finite_vals = log_vals[np.isfinite(log_vals)]
    if len(finite_vals) > 0:
        # Floor is min_observed - 2.0 (in log space)
        floor = finite_vals.min() - 2.0
    else:
        floor = -20.0  # Fallback if column is empty/all-zeros
        
    log_clipped = log_vals.copy()
    log_clipped[~np.isfinite(log_clipped)] = floor
    
    # C. Rolling Diurnal Adjustment
    # We calculate the Rolling Median of the LOG values for this specific time of day
    baseline = df.groupby(time_col)[col_name].transform(
        lambda x: log_clipped.loc[x.index].rolling(
            window=diurnal_window, min_periods=min_periods
        ).median().shift(1)  # Shift 1 to prevent leakage!
    )
    
    # Fill missing baselines (start of dataset)
    # If we don't have enough history for a baseline, assume baseline is the floor (or 0)
    baseline = baseline.fillna(method='bfill').fillna(floor)

    # D. Subtract Baseline (Log-Difference)
    adj_series = log_clipped - baseline
    
    return adj_series, baseline

def load_and_prep_data_strided(hparams, input_file):
    print(f"Loading {input_file}...")
    try: 
        data = pd.read_parquet(input_file, engine="pyarrow")
    except: 
        data = pd.read_csv(input_file)
        
    # 1. Standardize Time & Sort
    if 'endbartime' in data.columns: 
        data = data.rename(columns={'endbartime': 't', 'sumret2': 'RV'})
    
    data['t'] = pd.to_datetime(data['t'])
    
    # --- DEDUPLICATE (Critical for Reindexing) ---
    if data['t'].duplicated().any():
        print(f"Warning: Dropping {data['t'].duplicated().sum()} duplicate timestamps.")
        data = data.drop_duplicates(subset=['t'], keep='last')

    # Filter Dates
    data = data[data['t'] >= '2004-01-01']
    data = data[data['t'].dt.dayofweek < 5]
    
    # Align to full grid
    print("Aligning to full market grid...")
    dates = data['t'].dt.date.unique()
    all_slots = []
    for d in dates:
        day_slots = pd.date_range(start=f"{d} 00:00", end=f"{d} 23:30", freq="30min")
        all_slots.append(day_slots)
        
    full_grid = pd.DatetimeIndex(np.concatenate(all_slots)).sort_values()
    data = data.set_index('t').reindex(full_grid)
    data.index.name = 't'
    data = data.reset_index()

    # Fill Gaps & Winsorize (Standard Pre-processing)
    data['RV'] = data['RV'].fillna(0.0)
    
    w_window = hparams.get('winsor_window', 240) 
    rv_lower = data['RV'].rolling(window=w_window, min_periods=1).quantile(0.01)
    rv_upper = data['RV'].rolling(window=w_window, min_periods=1).quantile(0.99)
    data['RV'] = data['RV'].clip(lower=rv_lower, upper=rv_upper)

    data['time_of_day'] = data['t'].dt.time
    
    # 2. Process Target (RV) -> adj_log_RV
    print("Applying Robust Diurnal Adj to Target (RV)...")
    data['adj_log_RV'], data['baseline_RV'] = robust_log_diurnal_transform(data, 'RV', 'time_of_day')

    # ==============================================================================
    # --- NEW: STRICTLY SEGMENTED HAR FEATURES ---
    # ==============================================================================
    
    print("Generating Segment-Specific HAR Features...")
    
    # A. Define Segments
    minutes = data['t'].dt.hour * 60 + data['t'].dt.minute
    
    cond_morning = (minutes >= SEGMENT_THRESHOLDS['morning_start']) & (minutes < SEGMENT_THRESHOLDS['midday_start'])
    cond_midday  = (minutes >= SEGMENT_THRESHOLDS['midday_start']) & (minutes < SEGMENT_THRESHOLDS['closing_start'])
    cond_closing = (minutes >= SEGMENT_THRESHOLDS['closing_start']) & (minutes <= SEGMENT_THRESHOLDS['market_end'])
    
    data['segment'] = np.select(
        [cond_morning, cond_midday, cond_closing], 
        ['morning', 'midday', 'closing'], 
        default='overnight'
    )
    
    har_features = []
    
    # B. Generate Features per Segment
    segments = ['morning', 'midday', 'closing', 'overnight']
    
    for seg in segments:
        # Create a mask for the current segment
        seg_mask = (data['segment'] == seg)
        
        # Extract the target series ONLY for this segment
        seg_series = data.loc[seg_mask, 'adj_log_RV']
        
        for lag in HAR_LAGS:
            feat_name = f"har_{seg}_ma_{lag}"
            
            # 1. Initialize feature column with ZEROS (Strict Isolation)
            # This ensures 'morning' features are 0.0 when it is 'midday'
            data[feat_name] = 0.0
            
            # 2. Calculate Rolling Mean on the COMPRESSED series
            # (e.g., average of previous 5 'mornings')
            # shift(1) ensures we don't use the current morning to predict itself
            rolling_feat = seg_series.rolling(window=lag, min_periods=1).mean().shift(1)
            
            # 3. Place values back into the DataFrame ONLY at segment rows
            data.loc[seg_mask, feat_name] = rolling_feat
            
            # 4. Fill initial NaNs within the segment with 0
            data.loc[seg_mask, feat_name] = data.loc[seg_mask, feat_name].fillna(0.0)
            
            har_features.append(feat_name)

    # ==============================================================================
    # --- END NEW LOGIC ---
    # ==============================================================================

    # 3. Finalize Features
    final_cols = har_features 
    print(f"Final Features ({len(final_cols)}): {final_cols}")
    
    required_cols = ['t', 'segment', 'adj_log_RV', 'baseline_RV'] + final_cols
    data = data[required_cols]
    data = data.dropna()
    
    # Optional: Filter out 'overnight' rows if you only want to predict market hours
    # data = data[data['segment'] != 'overnight']
    
    print(f"Post-cleaning shape: {data.shape}")
    
    data = data.reset_index(drop=True)     
    
    X_np = data[final_cols].values.astype(np.float64)
    y_np = data['adj_log_RV'].values.astype(np.float64)
    dates = data['t']
    baselines = data['baseline_RV'].values
    
    return X_np, y_np, dates, baselines

# --- Keep existing helpers for Chunking/Saving ---
def get_chunk_indices_strided(X_np, train_window_size, chunk_id, total_chunks):
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
    y_subset = y_true[indices]
    
    # Handle dates: if it's a Series, use .iloc; if array, use direct indexing
    if hasattr(dates, 'iloc'):
        dates_subset = dates.iloc[indices].values
    else:
        dates_subset = dates[indices]
    
    base_subset = baselines[indices]
    
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