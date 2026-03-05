import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
DIURNAL_WINDOW = 20    # 20 Days
DIURNAL_MIN_PERIODS = 5 
# HAR Lags (Geometric Sequence)
HAR_LAGS = [1, 5, 25, 125, 625, 3125] 

def robust_sqrt_diurnal_transform(df, col_name, time_col="time_of_day", 
                                  diurnal_window=20, min_periods=5):
    """
    Applies Square Root transformations based on feature type.
    Includes safeguard for OBJECT/STRING columns.
    """
    
    # --- 1. DEFINE CATEGORIES ---
    SKIP_VARS = {'hour', 'DOW', 't', 'date'}
    SIGNED_KEYWORDS = ['sumret', 'autocov', 'sentiment', 'voldemand']
    MAGNITUDE_EXCEPTIONS = ['sumret2', 'sumret4', 'sumabsret', 'sumpret2']

    # --- 2. CLASSIFICATION LOGIC ---
    if col_name in SKIP_VARS:
        return df[col_name], pd.Series(0, index=df.index)

    is_signed = False
    for kw in SIGNED_KEYWORDS:
        if kw in col_name:
            if not any(ex in col_name for ex in MAGNITUDE_EXCEPTIONS):
                is_signed = True
                break
    
    # Force Numeric
    series_clean = pd.to_numeric(df[col_name], errors='coerce')

    if is_signed:
        print(f"  [Skipping Diurnal] '{col_name}' identified as Directional/Signed.")
        return series_clean.fillna(0.0), pd.Series(0.0, index=df.index)

    # --- 3. APPLY SQRT TRANSFORM ---
    
    # A. Square Root Transform
    # We clip at 0 to avoid imaginary numbers from noise/negative errors
    sqrt_vals = np.sqrt(series_clean.clip(lower=0))
    
    # B. Handling Missing Values
    # Instead of a log-floor, we just use 0 for the sqrt floor
    floor = 0.0
    sqrt_clipped = sqrt_vals.fillna(floor)
    
    # C. Rolling Diurnal Adjustment
    # Median is robust to outliers in the sqrt space
    baseline = df.groupby(time_col)[col_name].transform(
        lambda x: sqrt_clipped.loc[x.index].rolling(
            window=diurnal_window, min_periods=min_periods
        ).median().shift(1)
    )
    
    baseline = baseline.fillna(method='bfill').fillna(floor)
    adj_series = sqrt_clipped - baseline
    
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
    
    # Align to grid
    dates = data['t'].dt.date.unique()
    all_slots = []
    for d in dates:
        day_slots = pd.date_range(start=f"{d} 00:00", end=f"{d} 23:30", freq="30min")
        all_slots.append(day_slots)
        
    full_grid = pd.DatetimeIndex(np.concatenate(all_slots)).sort_values()
    data = data.set_index('t').reindex(full_grid)
    data.index.name = 't'
    data = data.reset_index()

    data['RV'] = data['RV'].fillna(0.0)
    
    exog_col_names = []
    if hparams.get("exog_cols") and str(hparams["exog_cols"]).lower() != "none":
        sep = '|' if '|' in hparams["exog_cols"] else ','
        raw_exog_list = hparams["exog_cols"].split(sep)
        exog_col_names = [c.strip() for c in raw_exog_list if c.strip() in data.columns]
        
        if exog_col_names:
            data[exog_col_names] = data[exog_col_names].fillna(method='ffill').fillna(0.0)

    data['time_of_day'] = data['t'].dt.time
    
    # 2. Process Target (RV) -> adj_sqrt_RV
    print("Applying Robust Diurnal Adj to Target (RV)...")
    data['adj_sqrt_RV'], data['baseline_RV'] = robust_sqrt_diurnal_transform(data, 'RV', 'time_of_day')
    
    # 3. Process Exogenous Features
    final_exog_feats = []
    for raw_col in exog_col_names:
        base_adj_col = f"adj_sqrt_{raw_col}"
        data[base_adj_col], _ = robust_sqrt_diurnal_transform(data, raw_col, 'time_of_day')
        
        for lag in HAR_LAGS:
            feat_name = f"{base_adj_col}_ma_{lag}"
            data[feat_name] = data[base_adj_col].rolling(window=lag).mean().shift(1)
            final_exog_feats.append(feat_name)

    # 4. Create HAR Features for Target (RV)
    har_features = []
    for lag in HAR_LAGS:
        feat_name = f"har_ma_{lag}"
        data[feat_name] = data['adj_sqrt_RV'].rolling(window=lag).mean().shift(1)
        har_features.append(feat_name)
        
    final_cols = har_features + final_exog_feats
    required_cols = ['t', 'adj_sqrt_RV', 'baseline_RV'] + final_cols
    data = data[required_cols].dropna().reset_index(drop=True) 
    
    X_np = data[final_cols].values.astype(np.float64)
    y_np = data['adj_sqrt_RV'].values.astype(np.float64)
    dates = data['t']
    baselines = data['baseline_RV'].values
    
    return X_np, y_np, dates, baselines

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

def save_chunk_results(output_file, forecasts, indices, y_true, dates, baselines):
    y_subset = y_true[indices]
    dates_subset = dates.iloc[indices].values if hasattr(dates, 'iloc') else dates[indices]
    base_subset = baselines[indices]
    
    # INVERSION LOGIC:
    # If y_sqrt = sqrt(RV) - baseline_sqrt, 
    # then RV = (y_sqrt + baseline_sqrt)^2
    
    # Note: Squaring a forecast is biased if there is error. 
    # Bias correction for sqrt: E[X] = (E[sqrt(X)])^2 + Var(sqrt(X))
    sigma2 = np.var(y_subset - forecasts)
    
    pred_raw = np.square(forecasts + base_subset) + sigma2/4
    true_raw = np.square(y_subset + base_subset)
    
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