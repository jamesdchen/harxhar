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
    
def calculate_baseline_deltas(summary_df):
    """
    Finds the baseline model and computes relative deltas and OOS R2.
    This keeps the comparative logic completely isolated from the file processing.
    """
    baseline_mask = (summary_df['exp_id'] == 0) | (summary_df['experiment_name'].str.lower().isin(['baseline', 'naive_baseline']))
    baseline_df = summary_df[baseline_mask]
    
    if baseline_df.empty:
        print("\n[Warning] No baseline experiment found. Deltas and OOS R2 will be NaN.")
        for col in ['delta_mse_raw', 'delta_mae_raw', 'delta_qlike', 'oos_r2']:
            summary_df[col] = np.nan
        return summary_df

    def get_baseline_val(segment_name, metric):
        b_row = baseline_df[baseline_df['segment'] == segment_name]
        return b_row.iloc[0].get(metric, np.nan) if not b_row.empty else np.nan

    # Vectorized comparisons
    summary_df['delta_mse_raw'] = summary_df.apply(lambda r: r['mse_raw'] - get_baseline_val(r['segment'], 'mse_raw'), axis=1)
    summary_df['delta_mae_raw'] = summary_df.apply(lambda r: r['mae_raw'] - get_baseline_val(r['segment'], 'mae_raw'), axis=1)
    summary_df['delta_qlike']   = summary_df.apply(lambda r: r['qlike']   - get_baseline_val(r['segment'], 'qlike'), axis=1)
    
    # OOS R2 Calculation (1 - MSE_model / MSE_baseline)
    summary_df['oos_r2'] = summary_df.apply(
        lambda r: 1.0 - (r['mse_raw'] / get_baseline_val(r['segment'], 'mse_raw')) 
        if get_baseline_val(r['segment'], 'mse_raw') > 0 else np.nan, 
        axis=1
    )

    return summary_df