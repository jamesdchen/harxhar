import numpy as np
import pandas as pd
import os
import argparse

# --- Import from your modular src/ directory ---
from src.data_tod import load_and_prep_data_strided
from src.data_helper import get_chunk_indices_strided, save_chunk_results
from src.models import RidgeModel, NaiveBaseline
from src.backtest import run_backtest_agnostic

def main(args):
    np.random.seed(42)
    
    hparams = {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
    }

    print(f"Loading segmented data from {args.input_path}...")
    
    # 1. LOAD DATA DICTIONARY 
    # Returns: {'morning': {'X': ..., 'y': ...}, 'midday': {...}, ...}
    datasets = load_and_prep_data_strided(hparams, args.input_path, target_segment='all')

    if not datasets:
        print("No datasets returned. Check data path and dates.")
        return

    # 2. Iterate over each Segment and Run Backtest
    for seg_name, data in datasets.items():
        print(f"\n" + "="*50)
        print(f"PROCESSING SEGMENT: {seg_name.upper()}")
        print("="*50)
        
        X_np = data['X']
        y_np = data['y']
        dates = data['dates']
        baselines = data['baselines']
        
        print(f"  Shape: X={X_np.shape}, Y={y_np.shape}")

        # --- DYNAMIC WINDOW CALCULATION ---
        # "Morning" has fewer slots per day than "Midday". 
        # We calculate the median slots per day to correctly size the training window.
        
        # Count rows per date
        daily_counts = dates.dt.date.value_counts()
        min_slots = daily_counts.min()
        max_slots = daily_counts.max()
        median_slots = int(daily_counts.median())
        
        print(f"  SEGMENT DIAGNOSTICS:")
        print(f"  - Min slots/day: {min_slots}")
        print(f"  - Max slots/day: {max_slots}")
        print(f"  - Median used:   {median_slots}")
                
        # Calculate window size in rows (Days * Slots/Day)
        train_win_periods = args.train_window * median_slots
        print(f"  Window size: {train_win_periods} rows ({args.train_window} days)")

        # Calculate Indices for this specific segment
        chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

        if chunk_idxs.size == 0: 
            print(f"  [Skipping] Chunk {args.chunk_id} empty for segment {seg_name}.")
            continue

        # --- INITIALIZE A MODULAR MODEL ---
        if args.model == 'ridge':
            print(f"  Initializing Ridge Model...")
            model = RidgeModel(
                train_win_periods=train_win_periods,
                n_features=X_np.shape[1],
                use_scaling=True,
                alpha=1.0 
            )
        elif args.model == 'naive':
            print(f"  Initializing Naive Baseline (Lag {args.naive_lag})...")
            model = NaiveBaseline(lag_index=args.naive_lag)
        elif args.model == 'xgboost':
            model = XGBoostModel(train_win_periods=args.train_window,
                 n_features=num_features,
                 use_scaling=False,
                 n_estimators=100,      # Add your preferred defaults
                 max_depth=3,           # Add your preferred defaults
                 learning_rate=0.1,
                 tree_method='hist')    # Highly recommended for speed
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        # --- RUN AGNOSTIC BACKTEST ---
        preds = run_backtest_agnostic(
            model=model,
            indices=chunk_idxs,
            X=X_np, 
            y=y_np, 
            train_win_periods=train_win_periods
        )

        # Extract the actual naive baseline for the evaluation script (instead of zeros)
        naive_preds = X_np[chunk_idxs, args.naive_lag]
        
        # --- SAVE RESULTS ---
        # Construct filename: output_file.csv -> output_file_morning.csv
        base, ext = os.path.splitext(args.output_file)
        seg_output_file = f"{base}_{seg_name}{ext}"
        
        save_chunk_results(
            output_file=seg_output_file, 
            forecasts=preds, 
            naive=naive_preds, 
            indices=chunk_idxs, 
            train_window=train_win_periods, 
            y_true=y_np, 
            dates=dates, 
            baselines=baselines
        )
        print(f"  Results saved to: {seg_output_file}")

    print("\nAll segments processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmented Time-Series Backtester")
    
    # Execution Routing
    parser.add_argument('--model', type=str, choices=['ridge', 'naive', 'xgboost'], required=True)
    parser.add_argument('--input-path', type=str, default="all30min")
    parser.add_argument('--output-file', type=str, required=True)
    
    # Chunking Options
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)

    # Model Params
    parser.add_argument('--train-window', type=int, default=500, help="Training window in DAYS")
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")    
    parser.add_argument('--naive-lag', type=int, default=0, help="Feature index for naive baseline")
    
    args = parser.parse_args()
    main(args)