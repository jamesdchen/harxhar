import os
import argparse
import re
import glob
import pandas as pd
import numpy as np

# Import from your modules
from src.eval_utils import load_all_chunks, parse_config, filter_by_date
from src.metrics import calculate_global_metrics

VALID_SEGMENTS = ['morning', 'midday', 'closing', 'overnight']

def main(args):
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    search_path = os.path.join(args.base_dir, "exp_*")
    exp_dirs = sorted(glob.glob(search_path), key=natural_sort_key)
    
    print("=" * 100)
    print(f"Found {len(exp_dirs)} experiments in '{args.base_dir}'")
    print(f"Mode: STRICT SEGMENTS ONLY {VALID_SEGMENTS}")
    if args.start_date:
        print(f"Filtering Results: AFTER {args.start_date}")
    print("=" * 100)
    
    results = []

    for exp_dir in exp_dirs:
        exp_id, exp_name = parse_config(exp_dir)
        print(f"Processing Exp {exp_id:<3} [{exp_name}]...", end=" ", flush=True)
        
        # --- THE MAGIC HAPPENS HERE ---
        # We tell the universal loader to ONLY grab segment files
        df = load_all_chunks(exp_dir, require_suffixes=VALID_SEGMENTS)
        
        if df.empty:
            print("[EMPTY / NO DATA]")
            continue
            
        original_len = len(df)
        df = filter_by_date(df, start_date=args.start_date)
        
        if df.empty:
            print(f"[EMPTY AFTER FILTER (Orig: {original_len})]")
            continue

        m = calculate_global_metrics(df)
        m['exp_id'] = exp_id
        m['experiment_name'] = exp_name
        results.append(m)
        
        print(f"[OK] n={m['n_samples']} | QLIKE (Filt): {m.get('qlike_filtered', np.nan):.4f} | MSE (Raw): {m.get('mse_raw', np.nan):.4e}")

    if not results:
        print("No valid results found. Exiting.")
        return
        
    summary_df = pd.DataFrame(results)
    baseline_row = summary_df[(summary_df['exp_id'] == 0) | (summary_df['experiment_name'].str.lower() == 'baseline') | (summary_df['experiment_name'].str.lower() == 'naive_baseline')]
    
    if not baseline_row.empty:
        base = baseline_row.iloc[0]
        summary_df['delta_qlike_filt'] = summary_df['qlike_filtered'] - base.get('qlike_filtered', np.nan)
        summary_df['delta_qlike_no'] = summary_df['qlike_nofilter'] - base.get('qlike_nofilter', np.nan)
        summary_df['delta_mse'] = summary_df['mse_adj'] - base.get('mse_adj', np.nan)
        summary_df['delta_mse_raw'] = summary_df['mse_raw'] - base.get('mse_raw', np.nan)
    else:
        for col in ['delta_qlike_filt', 'delta_qlike_no', 'delta_mse', 'delta_mse_raw']:
            summary_df[col] = np.nan

    summary_df = summary_df.sort_values('mse_raw' if 'mse_raw' in summary_df.columns else 'mse_adj')
    
    final_cols = [
        'exp_id', 'experiment_name', 
        'mse_raw', 'delta_mse_raw', 
        'mse_adj', 'delta_mse',
        'qlike_filtered', 'delta_qlike_filt', 
        'qlike_nofilter', 'delta_qlike_no',
        'n_samples'
    ]
    final_cols = [c for c in final_cols if c in summary_df.columns]
    
    print("\n" + "="*140)
    print("GLOBAL SUMMARY (Strict Segments Only - Sorted by Raw MSE)")
    print("="*140)
    
    pd.set_option('display.max_colwidth', 30)
    pd.set_option('display.width', 1000)
    
    formatters = {
        'mse_raw': '{:.4e}'.format,         
        'delta_mse_raw': '{:.4e}'.format, 
        'mse_adj': '{:.6f}'.format,         
        'delta_mse': '{:.6f}'.format,
        'qlike_filtered': '{:.6f}'.format,
        'delta_qlike_filt': '{:.6f}'.format,
    }
    
    print(summary_df[final_cols].to_string(index=False, formatters={k: v for k, v in formatters.items() if k in final_cols}))
    
    output_file = os.path.join(args.base_dir, "global_results_summary_segments.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate Global QLIKE & MSE (Segments)")
    parser.add_argument("--base_dir", type=str, default="results_ridge_subgroups")
    parser.add_argument("--start_date", type=str, default=None)
    main(parser.parse_args())