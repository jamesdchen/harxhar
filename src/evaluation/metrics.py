import numpy as np
import pandas as pd

def calculate_global_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculates MSE, MAE, and QLIKE (both filtered and non-filtered)."""
    metrics = {'n_samples': len(df)}
    
    # 1. Adjusted Scale Metrics
    if 'true_adj' in df.columns and 'pred_adj' in df.columns:
        metrics['mse'] = np.mean((df['true_adj'] - df['pred_adj'])**2)
        metrics['mae'] = np.mean(np.abs(df['true_adj'] - df['pred_adj']))
    
    # 2. Raw Scale Metrics
    if 'true_raw' in df.columns and 'pred_raw' in df.columns:
        mask_raw = (df['true_raw'] > 0) & (df['pred_raw'] > 0)
        if mask_raw.sum() > 0:
            vol_true = df.loc[mask_raw, 'true_raw']
            vol_pred = df.loc[mask_raw, 'pred_raw']
            metrics['qlike'] = np.mean((vol_true / vol_pred) - np.log(vol_true / vol_pred) - 1)
        else:
            metrics['qlike'] = np.nan
            
    return metrics
    
def calculate_baseline_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the baseline model and computes relative deltas and OOS R2.
    This keeps the comparative logic completely isolated from the file processing.
    """
    baseline_mask = (summary_df['exp_id'] == 0) | (summary_df['experiment_name'].str.lower().isin(['baseline', 'naive_baseline']))
    baseline_df = summary_df[baseline_mask]
    
    if baseline_df.empty:
        print("\n[Warning] No baseline experiment found. Deltas and OOS R2 will be NaN.")
        for col in ['delta_mse', 'delta_mae', 'delta_qlike', 'oos_r2']:
            summary_df[col] = np.nan
        return summary_df

    def get_baseline_val(segment_name, metric):
        b_row = baseline_df[baseline_df['segment'] == segment_name]
        return b_row.iloc[0].get(metric, np.nan) if not b_row.empty else np.nan

    # Vectorized comparisons
    summary_df['delta_mse'] = summary_df.apply(lambda r: r['mse'] - get_baseline_val(r['segment'], 'mse'), axis=1)
    summary_df['delta_mae'] = summary_df.apply(lambda r: r['mae'] - get_baseline_val(r['segment'], 'mae'), axis=1)
    summary_df['delta_qlike']   = summary_df.apply(lambda r: r['qlike']   - get_baseline_val(r['segment'], 'qlike'), axis=1)
    
    # OOS R2 Calculation (1 - MSE_model / MSE_baseline)
    summary_df['oos_r2'] = summary_df.apply(
        lambda r: 1.0 - (r['mse'] / get_baseline_val(r['segment'], 'mse')) 
        if get_baseline_val(r['segment'], 'mse') > 0 else np.nan, 
        axis=1
    )

    return summary_df