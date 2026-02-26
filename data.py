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
    
    # Apply the clip to prevent microscopic values from exploding the log
    series_clean = series_clean.clip(lower=1e-10)

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
    
    # 1. Load the data
    try: 
        data = pd.read_parquet(input_file, engine="pyarrow")
    except: 
        data = pd.read_csv(input_file)
        
    # 2. Standardize Time & Sort
    if 'endbartime' in data.columns:
        data = data.rename(columns={'endbartime': 't', 'sumret2': 'RV'})

    data['t'] = pd.to_datetime(data['t'])

    # Deduplicate before gridding
    if data['t'].duplicated().any():
        data = data.drop_duplicates(subset=['t'], keep='last')

    # 3. Define Global Boundaries
    start_date = "2005-01-01" 
    end_date = data['t'].max().date()
    
    full_grid = pd.date_range(start=f"{start_date} 00:00", 
                              end=f"{end_date} 23:30", 
                              freq="30min")
                              
    if len(full_grid) == 0: return pd.DataFrame()

    data = data.set_index('t').reindex(full_grid)
    data.index.name = 't'
    data = data.reset_index()

    # --- 3. Surgical Trimming (Drop planned closures) ---
    
    # Drop Friday after 20:00
    mask_friday_night = (data['t'].dt.dayofweek == 4) & (data['t'].dt.time > pd.to_datetime("20:00").time())
    
    # Drop all of Saturday (Monday=0, ..., Saturday=5)
    mask_saturday = data['t'].dt.dayofweek == 5
    
    # Drop Sunday before 18:30 (Sunday=6)
    mask_sunday_morning = (data['t'].dt.dayofweek == 6) & (data['t'].dt.time < pd.to_datetime("18:30").time())
    
    # Drop older data
    mask_pre_2007 = data['t'] < '2007-01-01'
    
    # Apply masks: Keep rows where ALL of these drop conditions are FALSE
    data = data[~(mask_friday_night | mask_saturday | mask_sunday_morning | mask_pre_2007)]

    # --- 4. Parse Exogenous Columns (if any) ---
    exog_col_names = []
    if hparams.get("exog_cols") and str(hparams["exog_cols"]).lower() != "none":
        sep = '|' if '|' in hparams["exog_cols"] else ','
        exog_col_names = [c.strip() for c in hparams["exog_cols"].split(sep) if c.strip() in data.columns]
        for col in exog_col_names:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # --- 5. Targeted Circuit Breaker Handling ---
    # The 4 days in 2020 where Level 1 market-wide circuit breakers tripped
    cb_dates = pd.to_datetime(['2020-03-09', '2020-03-12', '2020-03-16', '2020-03-18']).date
    mask_cb = data['t'].dt.date.isin(cb_dates) & (data['RV'] == 0.0)
    
    # Convert those exact 0.0s to NaN so pandas knows they are missing data
    data.loc[mask_cb, 'RV'] = np.nan

    # --- 6. True Missing Data Handling ---
    cols_to_fill = ['RV'] + exog_col_names
    
    # Forward-fill to bridge the circuit breakers and minor data drops (up to 2 periods / 1 hour)
    data[cols_to_fill] = data[cols_to_fill].ffill(limit=2)
    
    # Drop anything left over that couldn't be filled
    data = data.dropna(subset=cols_to_fill)
    
    # --- 6. Rolling Winsorization (1% - 99%) ---
    w_window = hparams.get('winsor_window', 240) 
    
    rv_lower = data['RV'].rolling(window=w_window, min_periods=1).quantile(0.01)
    rv_upper = data['RV'].rolling(window=w_window, min_periods=1).quantile(0.99)
    data['RV'] = data['RV'].clip(lower=rv_lower, upper=rv_upper)
    
    for col in exog_col_names:
        ex_lower = data[col].rolling(window=w_window, min_periods=1).quantile(0.01)
        ex_upper = data[col].rolling(window=w_window, min_periods=1).quantile(0.99)
        data[col] = data[col].clip(lower=ex_lower, upper=ex_upper)

    # --- 7. Log-Diurnal Transformation ---
    data['time_of_day'] = data['t'].dt.time
    data['adj_log_RV'], data['baseline_RV'] = robust_log_diurnal_transform(data, 'RV', 'time_of_day')
    
    final_exog_feats = []
    HAR_LAGS = [1, 5, 25, 125, 625, 3125]
    
    for raw_col in exog_col_names:
        base_adj_col = f"adj_log_{raw_col}"
        data[base_adj_col], _ = robust_log_diurnal_transform(data, raw_col, 'time_of_day')
        
        for lag in HAR_LAGS:
            feat_name = f"{base_adj_col}_ma_{lag}"
            data[feat_name] = data[base_adj_col].rolling(window=lag).mean().shift(1)
            final_exog_feats.append(feat_name)

    # --- 8. HAR Features for Target (RV) ---
    har_features = []
    for lag in HAR_LAGS:
        feat_name = f"har_ma_{lag}"
        data[feat_name] = data['adj_log_RV'].rolling(window=lag).mean().shift(1)
        har_features.append(feat_name)
        
    # --- 9. Final Clean & Matrix Extraction ---
    final_cols = har_features + final_exog_feats
    required_cols = ['t', 'adj_log_RV', 'baseline_RV'] + final_cols
    
    data = data[required_cols]
    
    # Drop rows with NaNs introduced by the longest HAR lags (the burn-in period)
    data = data.dropna().reset_index(drop=True)     
    
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