import pandas as pd
import numpy as np
import argparse
import os
import re
import glob

# --- CONFIGURATION ---
VALID_SEGMENTS = ['morning', 'midday', 'closing', 'overnight']

def load_all_chunks(exp_dir):
    """
    Loads result files that strictly end with a valid segment name.
    e.g., 'results_chunk_0_morning.csv' -> OK
          'results_chunk_0.csv'         -> IGNORE
    """
    # 1. Find all potential result files
    search_pattern = os.path.join(exp_dir, "results_chunk_*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        return pd.DataFrame()
    
    dfs = []
    
    # 2. Iterate and Filter
    for filename in all_files:
        # Check if file ends with a valid segment
        # We strip the extension (.csv) and check the suffix
        base_name = os.path.splitext(filename)[0] # remove .csv
        
        is_valid_segment = False
        for seg in VALID_SEGMENTS:
            if base_name.endswith(f"_{seg}"):
                is_valid_segment = True
                break
        
        if not is_valid_segment:
            # print(f"  [Skipping] {os.path.basename(filename)} (Not a segment file)")
            continue

        try:
            # Read CSV
            df = pd.read_csv(filename)
            
            # Standardize Date Index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            dfs.append(df)
        except Exception as e:
            pass
            
    if not dfs:
        return pd.DataFrame()
        
    # 3. Concatenate and Sort by Time
    full_df = pd.concat(dfs).sort_index()
    
    return full_df

def parse_config(exp_dir):
    """
    Reads the config.txt file to find the Experiment Name.
    """
    config_path = os.path.join(exp_dir, "config.txt")
    if not os.path.exists(config_path):
        return "Unknown"
    
    try:
        with open(config_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Experiment Name:"):
                    return line.split(":", 1)[1].strip()
            return "Unknown"
    except:
        pass
    return "Unknown"

def calculate_global_metrics(df):
    """
    Calculates GLOBAL metrics (MSE, QLIKE) on the aggregated dataframe.
    """
    metrics = {}
    
    # --- 1. MSE on Adjusted Target ---
    if 'true_adj' in df.columns and 'pred_adj' in df.columns:
        mse_adj = np.mean((df['true_adj'] - df['pred_adj'])**2)
        metrics['mse_adj'] = mse_adj
    
    # --- 2. Metrics on Raw Target ---
    if 'true_raw' in df.columns and 'pred_raw' in df.columns:
        
        # MSE Raw
        mse_raw = np.mean((df['true_raw'] - df['pred_raw'])**2)
        metrics['mse_raw'] = mse_raw

        # QLIKE: FILTERED (Expanding Min Epsilon)
        if len(df) > 100:
            current_min = df['true_raw'].iloc[:100].min()
        else:
            current_min = df['true_raw'].min()
            
        if current_min == 0 or np.isnan(current_min): current_min = 1e-5
        
        expanding_min = df['true_raw'].expanding().min()
        epsilon = expanding_min.shift(1).fillna(current_min) * 0.5
        
        mask_filt = (df['true_raw'] > epsilon) & (df['pred_raw'] > epsilon)
        
        if mask_filt.sum() > 0:
            vol_true = df.loc[mask_filt, 'true_raw']
            vol_pred = df.loc[mask_filt, 'pred_raw']
            ratio = vol_true / vol_pred
            log_diff = np.log(vol_true) - np.log(vol_pred)
            loss = ratio - log_diff - 1
            metrics['qlike_filtered'] = np.mean(loss)
        else:
            metrics['qlike_filtered'] = np.nan

        # QLIKE: NON-FILTERED
        mask_raw = (df['true_raw'] > 0) & (df['pred_raw'] > 0)

        if mask_raw.sum() > 0:
            vol_true = df.loc[mask_raw, 'true_raw']
            vol_pred = df.loc[mask_raw, 'pred_raw']
            ratio = vol_true / vol_pred
            log_diff = np.log(vol_true) - np.log(vol_pred) 
            loss = ratio - log_diff - 1
            metrics['qlike_nofilter'] = np.mean(loss)
        else:
            metrics['qlike_nofilter'] = np.nan
            
    metrics['n_samples'] = len(df)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate Global QLIKE & MSE (Strict Segments Only)")
    parser.add_argument("--base_dir", type=str, default="results_elasticnet_subgroups", help="Base directory containing exp_X folders")
    parser.add_argument("--start_date", type=str, default=None, help="Start date filter (YYYY-MM-DD).")

    args = parser.parse_args()

    # 1. Find all experiment directories
    search_path = os.path.join(args.base_dir, "exp_*")
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    exp_dirs = sorted(glob.glob(search_path), key=natural_sort_key)
    
    print(f"Found {len(exp_dirs)} experiments in {args.base_dir}")
    print(f"Valid Segments: {VALID_SEGMENTS}")
    if args.start_date:
        print(f"Filtering Results: Only including data AFTER {args.start_date}")
    print("-" * 100)
    
    results = []

    for exp_dir in exp_dirs:
        try:
            exp_id = int(exp_dir.split('_')[-1])
        except ValueError:
            exp_id = -1
            
        print(f"Processing Exp {exp_id:<3}...", end=" ", flush=True)
        
        exp_name = parse_config(exp_dir)
        print(f"[{exp_name}]...", end=" ", flush=True)
        
        # --- MODIFIED LOADING ---
        df = load_all_chunks(exp_dir)
        
        if df.empty:
            print("[EMPTY / MISSING]")
            continue

        # --- DATE FILTERING ---
        if args.start_date:
            try:
                start_ts = pd.Timestamp(args.start_date)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                original_len = len(df)
                df = df[df.index > start_ts]
                
                if df.empty:
                    print(f"[EMPTY AFTER FILTER (Orig: {original_len})]")
                    continue
            except Exception as e:
                print(f"[DATE ERROR: {e}]")
                continue

        # Calculate Metrics
        m = calculate_global_metrics(df)
        
        if m:
            m['exp_id'] = exp_id
            m['experiment_name'] = exp_name
            results.append(m)
            
            # Print quick status
            q_filt = f"{m.get('qlike_filtered', np.nan):.4f}"
            mse_adj_str = f"{m.get('mse_adj', np.nan):.4f}"
            print(f"[OK] n={m['n_samples']} | QLIKE: {q_filt} | MSE: {mse_adj_str}")
        else:
            print("[ERROR]")

    # 2. Aggregate and Sort
    if not results:
        print("No valid results found.")
        exit(1)
        
    summary_df = pd.DataFrame(results)
    
    # Calculate Deltas
    baseline_row = summary_df[summary_df['experiment_name'] == 'baseline']
    if baseline_row.empty:
        baseline_row = summary_df[summary_df['exp_id'] == 1]
    
    if not baseline_row.empty:
        base_qlike_filt = baseline_row.iloc[0].get('qlike_filtered', np.nan)
        base_qlike_no = baseline_row.iloc[0].get('qlike_nofilter', np.nan)
        base_mse_adj = baseline_row.iloc[0].get('mse_adj', np.nan)
        base_mse_raw = baseline_row.iloc[0].get('mse_raw', np.nan)
        
        summary_df['delta_qlike_filt'] = summary_df['qlike_filtered'] - base_qlike_filt
        summary_df['delta_qlike_no'] = summary_df['qlike_nofilter'] - base_qlike_no
        summary_df['delta_mse'] = summary_df['mse_adj'] - base_mse_adj
        summary_df['delta_mse_raw'] = summary_df['mse_raw'] - base_mse_raw
    else:
        for col in ['delta_qlike_filt', 'delta_qlike_no', 'delta_mse', 'delta_mse_raw']:
            summary_df[col] = np.nan

    # Sort
    if 'mse_raw' in summary_df.columns:
        summary_df = summary_df.sort_values('mse_raw')
    else:
        summary_df = summary_df.sort_values('mse_adj')
    
    # 3. Output Table
    cols = [
        'exp_id', 'experiment_name', 
        'mse_raw', 'delta_mse_raw', 
        'mse_adj', 'delta_mse',
        'qlike_filtered', 'delta_qlike_filt', 
        'qlike_nofilter', 'delta_qlike_no',
        'n_samples'
    ]
    
    final_cols = [c for c in cols if c in summary_df.columns]
    
    print("\n" + "="*160)
    print(f"GLOBAL SUMMARY (Strict Segments Only - Sorted by Raw MSE)")
    if args.start_date:
        print(f"Data Filter: AFTER {args.start_date}")
    print("="*160)
    
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 20)
    
    formatters = {
        'mse_raw': '{:.4e}'.format,         
        'delta_mse_raw': '{:.4e}'.format, 
        'mse_adj': '{:.6f}'.format,         
        'delta_mse': '{:.6f}'.format,
        'qlike_filtered': '{:.6f}'.format,
        'delta_qlike_filt': '{:.6f}'.format,
        'qlike_nofilter': '{:.6f}'.format,
        'delta_qlike_no': '{:.6f}'.format,
    }

    actual_formatters = {k: v for k, v in formatters.items() if k in final_cols}
    
    print(summary_df[final_cols].to_string(index=False, formatters=actual_formatters))
    
    output_file = os.path.join(args.base_dir, "global_results_summary.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")