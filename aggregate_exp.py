import os
import argparse
import re
import glob
import pandas as pd
import numpy as np

# Import the updated processor
from src.eval_utils import parse_config, process_single_experiment
from src.metrics import calculate_baseline_deltas


TARGET_SEGMENTS = ['morning', 'midday', 'closing', 'overnight']

# Define exact boundaries for the memory slicer
TOD_BOUNDS = {
    'morning':   {'start': '09:30', 'end': '11:30'},
    'midday':    {'start': '11:30', 'end': '14:00'},
    'closing':   {'start': '14:00', 'end': '16:00'},
    'overnight': {'start': '16:00', 'end': '09:30'}
}

def natural_sort_key(s):
    """Sorts strings containing numbers logically (e.g., exp_2 before exp_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main(args):
    # 1. --- Route Logic & Dynamic Configurations ---
    if args.eval_mode == 'segments':
        title_str = f"PRE-SEGMENTED FILES SUMMARY: {TARGET_SEGMENTS}"
        out_filename = "segment_results_summary.csv"
        segment_configs = [
            {"name": seg.upper(), "load_kwargs": {"require_suffixes": [seg], "ignore_suffixes": None}}
            for seg in TARGET_SEGMENTS
        ]
        
    elif args.eval_mode == 'filter_by_tod':
        title_str = "GLOBAL DATA (Filtered into TOD Segments in Memory)"
        out_filename = "global_results_tod_filtered.csv"
        # Create 4 configs that load global chunks, but slice them in memory
        segment_configs = [
            {
                "name": f"GLOBAL_{seg.upper()}", 
                "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS},
                "time_bounds": bounds
            }
            for seg, bounds in TOD_BOUNDS.items()
        ]
    else:
        title_str = "GLOBAL SUMMARY (All Hours)"
        out_filename = "global_results_summary.csv"
        segment_configs = [
            {"name": "GLOBAL", "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS}}
        ]
    # Find Experiments
    search_path = os.path.join(args.base_dir, "exp_*")
    exp_dirs = sorted(glob.glob(search_path), key=natural_sort_key)
    
    print("=" * 150)
    print(f"Found {len(exp_dirs)} experiments in '{args.base_dir}' | Mode: {args.eval_mode.upper()}")
    if args.eval_mode == 'filter_by_tod':
        print("Feature Active: Slicing global data by Time-of-Day (Morning, Midday, Closing, Overnight)")
    print("=" * 150)
   
    results = []

    # 2. --- Processing Loop ---
    for exp_dir in exp_dirs:
        exp_id, exp_name, model_type = parse_config(exp_dir)
        metadata = {
            'exp_id': exp_id,
            'experiment_name': exp_name,
            'model': model_type
        }
        
        # Call the agnostic processor imported from eval_utils
        results.extend(process_single_experiment(exp_dir, metadata, segment_configs))

    if not results:
        print("No valid results found. Exiting.")
        return
        
    # 3. --- Compile and Segment-Aware Delta Logic ---
    
    summary_df = calculate_baseline_deltas(pd.DataFrame(results))
    
    # 4. --- Final Table Output ---
    final_cols = [
        'exp_id', 'model', 'experiment_name', 'segment', 'cb_drop',
        'mse_raw', 'delta_mse_raw', 'oos_r2', 
        'mae_raw', 'delta_mae_raw',
        'qlike', 'delta_qlike', 'n_samples'
    ]
    final_cols = [c for c in final_cols if c in summary_df.columns]
    
    print("\n" + "="*175)
    print(title_str)
    print("="*165)
    
    formatters = {
        'mse_raw': '{:.4e}'.format,         
        'delta_mse_raw': '{:.4e}'.format, 
        'mae_raw': '{:.4e}'.format,         
        'delta_mae_raw': '{:.4e}'.format,
        'qlike': '{:.6f}'.format,
        'delta_qlike': '{:.6f}'.format,
        'oos_r2': '{:.4%}'.format,
        'cb_drop': lambda v: 'CB_DROP' if v else ''
    }
    
    pd.set_option('display.width', 1000)
    print(summary_df[final_cols].to_string(index=False, formatters={k: v for k, v in formatters.items() if k in final_cols}))
    
    output_file = os.path.join(args.base_dir, out_filename)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate Global/Segment Raw MSE, MAE & QLIKE")
    parser.add_argument("--eval-mode", type=str, choices=['global', 'segments', 'filter_by_tod'], default='global')
    parser.add_argument("--base_dir", type=str, default="results_ridge_subgroups")

    main(parser.parse_args())