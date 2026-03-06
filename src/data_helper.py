import pandas as pd
import numpy as np
from pathlib import Path
import os
from functools import reduce
from src import config

def robust_transform(df, col_name, time_col="time_of_day", 
                                 diurnal_window=config.DIURNAL_WINDOW, 
                                 min_periods=config.DIURNAL_MIN_PERIODS,
                                 use_log=True, allow_missing=False): # <-- NEW TOGGLE
    """Intelligently applies (optional) log and diurnal transformations."""
    SKIP_VARS = {'hour', 'DOW', 't', 'date'}
    SIGNED_KEYWORDS = ['sumret', 'autocov', 'sentiment', 'voldemand']
    MAGNITUDE_EXCEPTIONS = ['sumret2', 'sumret4', 'sumabsret', 'sumpret2']

    if col_name in SKIP_VARS:
        return df[col_name], pd.Series(0, index=df.index)

    is_signed = False
    for kw in SIGNED_KEYWORDS:
        if kw in col_name:
            if not any(ex in col_name for ex in MAGNITUDE_EXCEPTIONS):
                is_signed = True
                break
    
    series_clean = pd.to_numeric(df[col_name], errors='coerce')

    if is_signed:
        identity_baseline = 0.0 if use_log else 1.0
        filled = series_clean if allow_missing else series_clean.fillna(0.0)  # <-- CHANGE
        return filled, pd.Series(identity_baseline, index=df.index)
        
    # We clip at 0 if not logging, 1e-10 if logging
    series_clean = series_clean.clip(lower=1e-10 if use_log else 0.0)

    if use_log:
        with np.errstate(divide='ignore', invalid='ignore'):
            target_vals = np.log(series_clean)
            
        finite_vals = target_vals[np.isfinite(target_vals)]
        floor = finite_vals.min() - 2.0 if len(finite_vals) > 0 else -20.0
            
        target_clipped = target_vals.copy()
        target_clipped[~np.isfinite(target_clipped)] = floor
        
        baseline = df.groupby(time_col)[col_name].transform(
            lambda x: target_clipped.loc[x.index].rolling(
                window=diurnal_window, min_periods=min_periods
            ).median().shift(1)  
        )
        
        baseline = baseline.fillna(method='ffill').fillna(0.0)       
        
        # Log Space: Subtraction
        adj_series = target_clipped - baseline
    else:
        # Linear space: no log applied
        target_clipped = series_clean.copy()
        
        baseline = df.groupby(time_col)[col_name].transform(
            lambda x: target_clipped.loc[x.index].rolling(
                window=diurnal_window, min_periods=min_periods
            ).median().shift(1)  
        )
        
        baseline = baseline.fillna(method='ffill').fillna(1.0)
                
        baseline = baseline.clip(lower=1e-10) 
        
        # Linear Space: Division (Multiplicative Adjustment)
        adj_series = target_clipped / baseline
    
    return adj_series, baseline

def load_and_clean_base_data(hparams, input_path):
    """
    Handles stitching, gridding, trimming, and base log-diurnal transforms.
    Returns a clean DataFrame ready for HAR feature engineering.
    """
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith('.parquet')]
        dataframes = [pd.read_parquet(os.path.join(input_path, f), engine="pyarrow") for f in files]
        data = reduce(lambda left, right: pd.merge(left, right, on='endbartime', how='outer'), dataframes)
    else:
        data = pd.read_parquet(input_path, engine="pyarrow")
    
    if 'endbartime' in data.columns:
        data = data.rename(columns={'endbartime': 't', 'sumret2': 'RV'})

    data['t'] = pd.to_datetime(data['t'])
    if data['t'].duplicated().any():
        data = data.drop_duplicates(subset=['t'], keep='last')

    end_date = data['t'].max().date()
    full_grid = pd.date_range(start=f"{config.START_DATE} 00:00", end=f"{end_date} 23:30", freq="30min")
    if len(full_grid) == 0: return pd.DataFrame(), []

    data = data.set_index('t').reindex(full_grid)
    data.index.name = 't'  # Force the name back to 't'
    data = data.reset_index()

    # Drop non-trading hours
    mask_friday_night = (data['t'].dt.dayofweek == 4) & (data['t'].dt.time > pd.to_datetime("20:00").time())
    mask_saturday = data['t'].dt.dayofweek == 5
    mask_sunday_morning = (data['t'].dt.dayofweek == 6) & (data['t'].dt.time < pd.to_datetime("18:30").time())
    mask_pre_start = data['t'] < config.START_DATE
    data = data[~(mask_friday_night | mask_saturday | mask_sunday_morning | mask_pre_start)]

    exog_col_names = []
    if hparams.get("exog_cols") and str(hparams["exog_cols"]).lower() != "none":
        sep = '|' if '|' in hparams["exog_cols"] else ','
        exog_col_names = [c.strip() for c in hparams["exog_cols"].split(sep) if c.strip() in data.columns]
        for col in exog_col_names:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle Circuit Breakers and missing data
    cb_dates = pd.to_datetime(['2020-03-09', '2020-03-12', '2020-03-16', '2020-03-18']).date
    mask_cb = data['t'].dt.date.isin(cb_dates) & (data['RV'] == 0.0)
    data.loc[mask_cb, 'RV'] = np.nan

    allow_missing = hparams.get('allow_missing', False)

    # 1. ALWAYS clean the target variable (models need a y-value to train)
    data['RV'] = data['RV'].ffill()
    data = data.dropna(subset=['RV'])
    
    # 2. Conditionally clean the exogenous features
    if not allow_missing and exog_col_names:
        data[exog_col_names] = data[exog_col_names].ffill(limit=2)
        data = data.dropna(subset=exog_col_names)
    
    # Define columns to winsorize
    cols_to_fill = ['RV'] + exog_col_names
    
    # Winsorize (handling potential NaNs for XGBoost)
    w_window = hparams.get('winsor_window', 240) 
    for col in cols_to_fill:
        if allow_missing:
            # For XGBoost: use nanquantile so a single NaN doesn't void the whole window
            # min_periods=1 ensures we get a value as long as there is 1 non-NaN observation
            lower = data[col].rolling(window=w_window, min_periods=1).apply(
                lambda x: np.nanquantile(x, 0.01), raw=True
            )
            upper = data[col].rolling(window=w_window, min_periods=1).apply(
                lambda x: np.nanquantile(x, 0.99), raw=True
            )
        else:
            # For Ridge: standard quantile is faster since data is already ffilled/dropped
            lower = data[col].rolling(window=w_window, min_periods=1).quantile(0.01)
            upper = data[col].rolling(window=w_window, min_periods=1).quantile(0.99)
            
        data[col] = data[col].clip(lower=lower, upper=upper)

    # Transforms
    use_log = hparams.get('use_log', True)
    prefix = "adj_log_" if use_log else "adj_"

    data['time_of_day'] = data['t'].dt.time
    
    # Target transform
    target_col_name = f"{prefix}RV"
    data[target_col_name], data['baseline_RV'] = robust_transform(
        data, 'RV', 'time_of_day', use_log=use_log, allow_missing=allow_missing
    )
    
    # THE FIX: Dynamically track the target column name
    cols_to_transform = [target_col_name]

    for raw_col in exog_col_names:
        base_adj_col = f"{prefix}{raw_col}"
        data[base_adj_col], _ = robust_transform(
            data, raw_col, 'time_of_day', use_log=use_log, allow_missing=allow_missing
        )

    return data, cols_to_transform

def get_chunk_indices_strided(X_np, train_window_size, chunk_id, total_chunks):
    """Calculates indices for chunked evaluation."""
    num_samples = X_np.shape[0]
    valid_test_start = train_window_size
    if valid_test_start >= num_samples: return np.array([])
    test_indices = np.arange(valid_test_start, num_samples)
    chunk_indices_list = np.array_split(test_indices, total_chunks)
    if chunk_id >= len(chunk_indices_list): return np.array([])
    return chunk_indices_list[chunk_id]

def save_chunk_results(output_file, forecasts, indices, train_window, y_true, dates, baselines, use_log=True):
    """Saves predictions and reconstructs raw space values for the primary model only."""
    y_subset = y_true[indices]
    base_subset = baselines[indices]
    dates_subset = dates.iloc[indices].values if hasattr(dates, 'iloc') else dates[indices]
    
    if use_log:
        # Reconstruct from Log Space using Duan's Smearing for the model
        sigma2_model = np.var(y_subset - forecasts)
        pred_raw = np.exp(forecasts + base_subset + (sigma2_model / 2))
        true_raw = np.exp(y_subset + base_subset)
    else:
        # Reconstruct from Linear Space (Diurnal baseline back-multiplication)
        pred_raw = forecasts * base_subset
        true_raw = y_subset * base_subset
    
    # DataFrame now only contains true vs. model predicted
    df = pd.DataFrame({
        'date': dates_subset,
        'true_adj': y_subset,
        'pred_adj': forecasts,     
        'true_raw': true_raw,
        'pred_raw': pred_raw
    })
    
    from pathlib import Path
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return dates_subset