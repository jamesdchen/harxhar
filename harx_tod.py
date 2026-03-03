import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from tqdm import tqdm
import argparse

# Assuming these exist in your environment
from rolling import RollingStandardScaler, RollingBuffer
from data_tod import load_and_prep_data_strided, get_chunk_indices_strided, save_chunk_results

def run_backtest_pooled_rolling(indices, X, y, train_win_periods, use_scaling=True):
    """
    Runs a Pooled Rolling backtest on the flattened data.
    """
    n_features = X.shape[1]
    n_targets = 1

    # Initialize Helpers
    scaler_x = RollingStandardScaler(n_features)
    buffer = RollingBuffer(train_win_periods, n_features, n_targets)

    # --- 1. Pre-Fill Buffer (Training History) ---
    first_test_idx = indices[0]

    if first_test_idx < train_win_periods:
        # In the segmented approach, chunks might be smaller, so we handle this gracefully
        # or raise error if critical.
        raise ValueError(f"Not enough history. Test starts at idx {first_test_idx}, but need {train_win_periods} periods.")

    # Slice initial training data
    start_hist = first_test_idx - train_win_periods
    X_init = X[start_hist : first_test_idx]
    y_init = y[start_hist : first_test_idx]

    if y_init.ndim == 1:
        y_init = y_init.reshape(-1, 1)

    # Initialize Scaler (X only)
    if use_scaling:
        scaler_x.initialize(X_init)
        mean_x, std_x = scaler_x.get_scaler()
    else:
        mean_x = np.zeros(n_features)
        std_x = np.ones(n_features)

    # Fill Buffer
    buffer.X_buffer[:] = (X_init - mean_x) / std_x
    buffer.y_buffer[:] = y_init 

    # Keep raw history for online updates
    hist_X = list(X_init)
    hist_y = list(y_init)

    # Initial Fit
    model = Ridge() 
    X_tr, y_tr = buffer.get_view()
    model.fit(X_tr, y_tr)

    # Output arrays
    n_preds = len(indices)
    preds = np.zeros(n_preds)

    # --- 2. Rolling Loop (Period by Period) ---
    for i, t_idx in enumerate(indices):

        # A. Predict current step t
        x_target_raw = X[t_idx] 

        # Scale X using current rolling stats
        x_scl = (x_target_raw - mean_x) / std_x

        # Predict
        pred = model.predict(x_scl.reshape(1, -1))
        preds[i] = pred.item()

        # B. Update Model with realized value at t (Walk-Forward)
        y_realized = y[t_idx]

        # Pop Oldest
        x_old = hist_X.pop(0)
        y_old = hist_y.pop(0)

        # Update Scaler
        if use_scaling:
            scaler_x.update(x_target_raw, x_old)
            mean_x, std_x = scaler_x.get_scaler()

        # Push New Data to Buffer
        x_new_scl = (x_target_raw - mean_x) / std_x
        buffer.add(x_new_scl, y_realized)

        # Update raw history lists
        hist_X.append(x_target_raw)
        hist_y.append(y_realized)

        # Refit
        X_tr, y_tr = buffer.get_view()
        model.fit(X_tr, y_tr)

    return preds

def main(args):
    np.random.seed(42)
    
    hparams = {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
    }

    print("Loading data...")
    # 1. LOAD DATA DICTIONARY (Modified return from data_tod.py)
    # Returns: {'morning': {...}, 'midday': {...}, ...}
    datasets = load_and_prep_data_strided(hparams, "all30min")

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
        median_slots = int(daily_counts.median())
        # ... inside your loop ...
        min_slots = daily_counts.min()
        max_slots = daily_counts.max()
        
        print(f"  SEGMENT DIAGNOSTICS:")
        print(f"  - Min slots/day: {min_slots}")
        print(f"  - Max slots/day: {max_slots}")
        print(f"  - Median used:   {median_slots}")
                
        # Calculate window size in rows (Days * Slots/Day)
        train_win_periods = args.train_window * median_slots
        print(f"  Median slots/day: {median_slots} | Window size: {train_win_periods} rows ({args.train_window} days)")

        # Calculate Indices for this specific segment
        chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

        if chunk_idxs.size == 0: 
            print(f"  [Skipping] Chunk empty for segment {seg_name}.")
            continue

        # Run Backtest
        preds = run_backtest_pooled_rolling(
            chunk_idxs,
            X_np, 
            y_np, 
            train_win_periods
        )

        # Save Results (Separate File per Segment)
        dummy_naive = np.zeros_like(preds)
        
        # Construct filename: output_file.csv -> output_file_morning.csv
        base, ext = os.path.splitext(args.output_file)
        seg_output_file = f"{base}_{seg_name}{ext}"
        
        save_chunk_results(
            seg_output_file, 
            preds, 
            dummy_naive, 
            chunk_idxs, 
            train_win_periods, 
            y_np, 
            dates, 
            baselines
        )
        print(f"  Results saved to: {seg_output_file}")

    print("\nAll segments processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)

    # Model Params
    parser.add_argument('--train-window', type=int, default=500, help="Training window in DAYS")
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")    
    args = parser.parse_args()
    main(args)