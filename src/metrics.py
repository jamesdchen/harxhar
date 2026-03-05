import numpy as np
import pandas as pd

def calculate_global_metrics(df):
    """Calculates MSE, MAE, and QLIKE (both filtered and non-filtered)."""
    metrics = {'n_samples': len(df)}
    
    # 1. Adjusted Scale Metrics (Log Space)
    if 'true_adj' in df.columns and 'pred_adj' in df.columns:
        metrics['mse_adj'] = np.mean((df['true_adj'] - df['pred_adj'])**2)
        metrics['mae_adj'] = np.mean(np.abs(df['true_adj'] - df['pred_adj']))
    
    # 2. Raw Scale Metrics
    if 'true_raw' in df.columns and 'pred_raw' in df.columns:
        metrics['mse_raw'] = np.mean((df['true_raw'] - df['pred_raw'])**2)
        metrics['mae_raw'] = np.mean(np.abs(df['true_raw'] - df['pred_raw']))

        # --- QLIKE: FILTERED ---
        current_min = df['true_raw'].iloc[:100].min() if len(df) > 100 else df['true_raw'].min()
        if current_min == 0 or np.isnan(current_min): 
            current_min = 1e-5
            
        expanding_min = df['true_raw'].expanding().min()
        epsilon = expanding_min.shift(1).fillna(current_min) * 0.5
        
        mask_filt = (df['true_raw'] > epsilon) & (df['pred_raw'] > epsilon)
        
        if mask_filt.sum() > 0:
            vol_true = df.loc[mask_filt, 'true_raw']
            vol_pred = df.loc[mask_filt, 'pred_raw']
            metrics['qlike_filtered'] = np.mean((vol_true / vol_pred) - np.log(vol_true / vol_pred) - 1)
        else:
            metrics['qlike_filtered'] = np.nan

        # --- QLIKE: NON-FILTERED ---
        mask_raw = (df['true_raw'] > 0) & (df['pred_raw'] > 0)
        if mask_raw.sum() > 0:
            vol_true = df.loc[mask_raw, 'true_raw']
            vol_pred = df.loc[mask_raw, 'pred_raw']
            metrics['qlike_nofilter'] = np.mean((vol_true / vol_pred) - np.log(vol_true / vol_pred) - 1)
        else:
            metrics['qlike_nofilter'] = np.nan
            
    return metrics