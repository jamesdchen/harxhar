import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
DIURNAL_WINDOW = 20    
DIURNAL_MIN_PERIODS = 5 
HAR_LAGS = [1, 5, 25, 125, 625, 3125] 

# 1. Define Segments with Overlaps
SEGMENT_DEFINITIONS = {
    'morning':   {'start': 510, 'end': 660},   # 08:30 - 11:00
    'midday':    {'start': 630, 'end': 870},   # 10:30 - 14:30
    'closing':   {'start': 840, 'end': 960},   # 14:00 - 16:00
    'overnight': {'start': 990, 'end': 510}    # 16:30 - 08:30 (Wraps)
}

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
        
    if 'endbartime' in data.columns: 
        data = data.rename(columns={'endbartime': 't', 'sumret2': 'RV'})
    data['t'] = pd.to_datetime(data['t'])
    
    # --- Deduplicate (Critical) ---
    if data['t'].duplicated().any():
        print(f"Warning: Dropping {data['t'].duplicated().sum()} duplicate timestamps.")
        data = data.drop_duplicates(subset=['t'], keep='last')
        
    # Filter Dates
    data = data[data['t'] >= '2010-01-01']
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

    # Fill & Winsorize
    data['RV'] = data['RV'].fillna(0.0)
    w_window = hparams.get('winsor_window', 240) 
    rv_lower = data['RV'].rolling(window=w_window, min_periods=1).quantile(0.01)
    rv_upper = data['RV'].rolling(window=w_window, min_periods=1).quantile(0.99)
    data['RV'] = data['RV'].clip(lower=rv_lower, upper=rv_upper)

    data['time_of_day'] = data['t'].dt.time
    
    # Process Target
    print("Applying Robust Diurnal Adj to Target (RV)...")
    data['adj_log_RV'], data['baseline_RV'] = robust_log_diurnal_transform(data, 'RV', 'time_of_day')

    # ==============================================================================
    # --- NEW: RETURN DICTIONARY OF DATASETS ---
    # ==============================================================================
    
    print("Splitting Data into Segment-Specific Datasets...")
    
    # 1. Identify cols to transform (Target + Exog)
    cols_to_transform = ['adj_log_RV']
    
    # Process Exogenous Cols (Global transform first)
    exog_col_names = []
    if hparams.get("exog_cols") and str(hparams["exog_cols"]).lower() != "none":
        sep = '|' if '|' in hparams["exog_cols"] else ','
        raw_exog_list = hparams["exog_cols"].split(sep)
        exog_col_names = [c.strip() for c in raw_exog_list if c.strip() in data.columns]
        
        for col in exog_col_names:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(method='ffill').fillna(0.0)
            base_adj_col = f"adj_log_{col}"
            data[base_adj_col], _ = robust_log_diurnal_transform(data, col, 'time_of_day')
            cols_to_transform.append(base_adj_col)

    minutes = data['t'].dt.hour * 60 + data['t'].dt.minute
    
    # This dictionary will store the result: {'morning': (X, y, dates, base), ...}
    datasets = {}

    for seg_name, times in SEGMENT_DEFINITIONS.items():
        start = times['start']
        end = times['end']
        
        # A. Create Mask
        if start < end:
            mask = (minutes >= start) & (minutes <= end)
        else:
            mask = (minutes >= start) | (minutes <= end)
            
        # B. Extract Data for this Segment
        # We work on a COPY so calculations don't bleed into other segments
        seg_df = data.loc[mask].copy()
        
        if seg_df.empty:
            print(f"Warning: Segment {seg_name} is empty. Skipping.")
            continue
            
        print(f"Processing Segment: {seg_name} (Rows: {len(seg_df)})")

        segment_features = []
        
        # C. Generate Features strictly on this subset
        for col in cols_to_transform:
            for lag in HAR_LAGS:
                feat_name = f"{col}_ma_{lag}" # Name is generic now, implicitly "this segment"
                
                # Calculate Rolling Mean on the COMPRESSED series
                # Shift(1) ensures prediction for T uses only T-1 backwards (within segment)
                seg_df[feat_name] = seg_df[col].rolling(window=lag, min_periods=1).mean().shift(1)
                
                segment_features.append(feat_name)
        
        # D. Clean this specific dataset
        # Drop rows where lag creation caused NaNs (the burn-in period)
        seg_df = seg_df.dropna()
        
        if seg_df.empty:
            print(f"Warning: Segment {seg_name} is empty after dropping NaNs.")
            continue

        # E. Pack Results
        X_seg = seg_df[segment_features].values.astype(np.float64)
        y_seg = seg_df['adj_log_RV'].values.astype(np.float64)
        dates_seg = seg_df['t']
        base_seg = seg_df['baseline_RV'].values
        
        datasets[seg_name] = {
            'X': X_seg,
            'y': y_seg,
            'dates': dates_seg,
            'baselines': base_seg,
            'features': segment_features
        }

    return datasets

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