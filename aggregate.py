import pandas as pd
import numpy as np
import argparse
import os

def load_all_chunks(file_pattern, num_files):
    """
    Loads all result chunks and concatenates them into a single DataFrame.
    """
    dfs = []
    print(f"Loading {num_files} chunks...")
    
    for i in range(1, num_files + 1):
        filename = file_pattern.format(i)
        
        if not os.path.exists(filename):
            print(f"  Warning: {filename} not found. Skipping.")
            continue
            
        try:
            # Flattened CSVs have the timestamp in the first column (index 0).
            # We parse it as the index.
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    # Combine and sort by time to stitch disjoint chunks together
    full_df = pd.concat(dfs).sort_index()
    return full_df

def calculate_intraday_metrics(df):
    """
    Calculates Intraday metrics matching 'Volatility Forecasting with Machine Learning' (Zhang et al.).
    Reflects the continuous nature of the model (no daily aggregation).
    """
    required_cols = ['true_raw', 'pred_raw']
    if not all(col in df.columns for col in required_cols):
        print("Missing required columns (true, pred).")
        return None

    # Determine if we have naive column
    has_naive = 'naive' in df.columns
    
    # Copy to avoid side effects
    df = df.copy()

    # --- Loss Functions (Defined for reuse) ---
    def paper_mse(vol_true, vol_pred):
        # MSE on Logs (Eq 4)
        return np.mean((np.log(vol_true) - np.log(vol_pred))**2)

    def paper_qlike(vol_true, vol_pred):
        # QLIKE: ratio - log(ratio) - 1
        ratio = vol_true / vol_pred
        log_diff = np.log(vol_true) - np.log(vol_pred)
        loss = ratio - log_diff - 1
        return np.mean(loss)

    # 1. Intraday Metrics (30-min)
    # ---------------------------------------------------------
    current_min = df['true_raw'].iloc[:100].min()
    
    # Safety:
    if current_min == 0 or np.isnan(current_min): current_min = 1e-5
    
    print(f"Initial Floor (First 100 obs): {current_min:.2e}")
    
    # 2. Iterate through the timeline
    # Vectorized approach: "Expanding Min"
    
    # Calculate the running minimum of the 'true_raw' column
    # This represents "The lowest volatility observed so far in history"
    expanding_min = df['true_raw'].expanding().min()
    
    # Shift it by 1! We can only use the min up to t-1 to clip prediction at t.
    # We use a safety buffer (e.g., 0.5 * historical_min)
    epsilon = expanding_min.shift(1).fillna(current_min) * 0.5
    
    # Filter strictly valid data (no zeros/negatives allowed in logs)
    mask = (df['true_raw'] > epsilon) & (df['pred_raw'] > epsilon)
    if has_naive: mask = mask & (df['naive'] > epsilon)
    
    df_intra = df[mask].copy()
    
    metrics = {}
    
    metrics['intraday_mse'] = paper_mse(df_intra['true_raw'], df_intra['pred_raw'])
    metrics['intraday_qlike'] = paper_qlike(df_intra['true_raw'], df_intra['pred_raw'])
    
    if has_naive:
        metrics['intraday_mse_naive'] = paper_mse(df_intra['true_raw'], df_intra['naive'])
        metrics['intraday_qlike_naive'] = paper_qlike(df_intra['true_raw'], df_intra['naive'])

    metrics['n_samples_intraday'] = len(df_intra)
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Intraday MSE/QLIKE matching Zhang et al. (2023)")
    parser.add_argument("--num_files", type=int, required=True, help="Total chunks.")
    parser.add_argument("--file_pattern", type=str, default="results_best_model/results_chunk_{}.csv")
    
    args = parser.parse_args()

    full_df = load_all_chunks(args.file_pattern, args.num_files)
    
    if full_df.empty:
        print("No data loaded.")
        exit(1)

    m = calculate_intraday_metrics(full_df)
    
    if m:
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS | Intraday Samples: {m['n_samples_intraday']}")
        print("="*60)
        
        # Helper to print row
        def print_row(label, val_model, val_naive=None):
            naive_str = f" | Naive: {val_naive:.6f}" if val_naive is not None else ""
            print(f"{label:<20} : {val_model:.6f}{naive_str}")

        print("\n--- Intraday (Continuous) ---")
        print_row("MSE (on logs)", m['intraday_mse'], m.get('intraday_mse_naive'))
        print_row("QLIKE", m['intraday_qlike'], m.get('intraday_qlike_naive'))
        print("="*60 + "\n")