import os
import argparse
import re
import glob
import pandas as pd
import numpy as np

# Import from your modules
from src.eval_utils import load_all_chunks, parse_config, filter_by_date
from src.metrics import calculate_global_metrics

# Define the core segments once
TARGET_SEGMENTS = ['morning', 'midday', 'closing', 'overnight']

def natural_sort_key(s):
    """Sorts strings containing numbers logically (e.g., exp_2 before exp_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main(args):
    # 1. --- Route Logic Based on Mode ---
    if args.eval_mode == 'segments':
        req_suffixes = TARGET_SEGMENTS
        ign_suffixes = None
        title_str = f"GLOBAL SUMMARY (Strict Segments Only: {TARGET_SEGMENTS})"
        out_filename = "global_results_summary_segments.csv"
    else: # 'global'
        req_suffixes = None
        ign_suffixes = TARGET_SEGMENTS
        title_str = "GLOBAL SUMMARY (Sorted by Raw MSE)"
        date_suffix = "_filtered" if (args.start_date or args.end_date) else ""
        out_filename = f"global_results_summary{date_suffix}.csv"

    # 2. --- Find Experiments ---
    search_path = os.path.join(args.base_dir, "exp_*")
    exp_dirs = sorted(glob.glob(search_path), key=natural_sort_key)
    
    print("=" * 100)
    print(f"Found {len(exp_dirs)} experiments in '{args.base_dir}'")
    print(f"Mode: {args.eval_mode.upper()}")
    
    if args.start_date or args.end_date:
        start_str = args.start_date if args.start_date else "Beginning"
        end_str = args.end_date if args.end_date else "End"
        print(f"Filtering Results: {start_str} to {end_str}")
    print("=" * 100)
    
    results = []

    # 3. --- Processing Loop ---
    for exp_dir in exp_dirs:
        exp_id, exp_name = parse_config(exp_dir)
        print(f"Processing Exp {exp_id:<3} [{exp_name}]...", end=" ", flush=True)
        
        # Load exactly what is requested based on the mode
        df = load_all_chunks(
            exp_dir, 
            require_suffixes=req_suffixes, 
            ignore_suffixes=ign_suffixes
        )
        
        if df.empty:
            print("[EMPTY / NO DATA]")
            continue
            
        original_len = len(df)
        df = filter_by_date(df, start_date=args.start_date, end_date=args.end_date)
        
        if df.empty:
            print(f"[EMPTY AFTER FILTER (Orig: {original_len})]")
            continue

        m = calculate_global_metrics(df)
        m['exp_id'] = exp_id
        m['experiment_name'] = exp_name
        results.append(m)
        
        # --- NEW: Print MAE in the live console output ---
        print(f"[OK] n={m['n_samples']} | QLIKE (Filt): {m.get('qlike_filtered', np.nan):.4f} | MSE: {m.get('mse_raw', np.nan):.4e} | MAE: {m.get('mae_raw', np.nan):.4e}")

    if not results:
        print("No valid results found. Exiting.")
        return
        
    # 4. --- Compile and Compare to Baseline ---
    summary_df = pd.DataFrame(results)
    
    # Check for either boolean True (if boolean) or string matches
    if summary_df['experiment_name'].dtype == object:
        baseline_mask = (summary_df['exp_id'] == 0) | (summary_df['experiment_name'].str.lower().isin(['baseline', 'naive_baseline']))
    else:
        baseline_mask = summary_df['exp_id'] == 0
        
    baseline_row = summary_df[baseline_mask]
    
    if not baseline_row.empty:
        base = baseline_row.iloc[0]
        summary_df['delta_qlike_filt'] = summary_df['qlike_filtered'] - base.get('qlike_filtered', np.nan)
        summary_df['delta_qlike_no'] = summary_df['qlike_nofilter'] - base.get('qlike_nofilter', np.nan)
        summary_df['delta_mse'] = summary_df['mse_adj'] - base.get('mse_adj', np.nan)
        summary_df['delta_mse_raw'] = summary_df['mse_raw'] - base.get('mse_raw', np.nan)
        # --- NEW: Calculate MAE Deltas ---
        summary_df['delta_mae'] = summary_df['mae_adj'] - base.get('mae_adj', np.nan)
        summary_df['delta_mae_raw'] = summary_df['mae_raw'] - base.get('mae_raw', np.nan)
    else:
        print("\n[Warning] No baseline experiment found (ID 0 or Name 'baseline'). Delta columns will be NaN.")
        for col in ['delta_qlike_filt', 'delta_qlike_no', 'delta_mse', 'delta_mse_raw', 'delta_mae', 'delta_mae_raw']:
            summary_df[col] = np.nan

    summary_df = summary_df.sort_values('mse_raw' if 'mse_raw' in summary_df.columns else 'mse_adj')
    
    # --- NEW: Added MAE columns to the final output list ---
    final_cols = [
        'exp_id', 'experiment_name', 
        'mse_raw', 'delta_mse_raw', 
        'mae_raw', 'delta_mae_raw',
        'mse_adj', 'delta_mse',
        'mae_adj', 'delta_mae',
        'qlike_filtered', 'delta_qlike_filt', 
        'qlike_nofilter', 'delta_qlike_no',
        'n_samples'
    ]
    final_cols = [c for c in final_cols if c in summary_df.columns]
    
    # 5. --- Formatting and Output ---
    print("\n" + "="*140)
    print(title_str)
    print("="*140)
    
    pd.set_option('display.max_colwidth', 30)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    
    # --- NEW: Formatting rules for MAE ---
    formatters = {
        'mse_raw': '{:.4e}'.format,         
        'delta_mse_raw': '{:.4e}'.format, 
        'mae_raw': '{:.4e}'.format,         
        'delta_mae_raw': '{:.4e}'.format,
        'mse_adj': '{:.6f}'.format,         
        'delta_mse': '{:.6f}'.format,
        'mae_adj': '{:.6f}'.format,         
        'delta_mae': '{:.6f}'.format,
        'qlike_filtered': '{:.6f}'.format,
        'delta_qlike_filt': '{:.6f}'.format,
        'qlike_nofilter': '{:.6f}'.format,
        'delta_qlike_no': '{:.6f}'.format,
    }
    
    print(summary_df[final_cols].to_string(index=False, formatters={k: v for k, v in formatters.items() if k in final_cols}))
    
    output_file = os.path.join(args.base_dir, out_filename)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate Global QLIKE, MSE, & MAE")
    
    parser.add_argument("--eval-mode", type=str, choices=['global', 'segments'], default='global',
                        help="Choose whether to evaluate global chunks or specific segments.")
    parser.add_argument("--base_dir", type=str, default="results_ridge_subgroups", help="Base directory containing exp_X folders")
    parser.add_argument("--start_date", type=str, default=None, help="Start date filter (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=None, help="End date filter (YYYY-MM-DD).")

    main(parser.parse_args())