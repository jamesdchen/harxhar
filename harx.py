import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from tqdm import tqdm
import argparse

# Assuming these exist in your environment
from rolling import RollingStandardScaler, RollingBuffer
from data import load_and_prep_data_strided, get_chunk_indices_strided, save_chunk_results

def run_backtest_pooled_rolling(indices, X, y, train_win_periods, use_scaling=True):
    """
    Runs a Pooled Rolling backtest on the flattened data.
    """
    n_features = X.shape[1]
    n_targets = 1

    # Initialize Helpers
    # We use RollingStandardScaler for X (Z-Score)
    scaler_x = RollingStandardScaler(n_features)

    # We use RollingBuffer to hold the training window
    buffer = RollingBuffer(train_win_periods, n_features, n_targets)

    # --- 1. Pre-Fill Buffer (Training History) ---
    first_test_idx = indices[0]

    if first_test_idx < train_win_periods:
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
        # Get Mean and Std from the new RollingStandardScaler
        mean_x, std_x = scaler_x.get_scaler()
    else:
        mean_x = np.zeros(n_features)
        std_x = np.ones(n_features)

    # Fill Buffer
    # Scale X using (X - Mean) / Std
    # Leave Y raw (No scaling needed for adjusted target)
    buffer.X_buffer[:] = (X_init - mean_x) / std_x
    buffer.y_buffer[:] = y_init 

    # Keep raw history for online updates
    hist_X = list(X_init)
    hist_y = list(y_init)

    # Initial Fit
    # Use Ridge for stability with many exog features
    model = Ridge() #ElasticNet() #LinearRegression()
    X_tr, y_tr = buffer.get_view()
    model.fit(X_tr, y_tr)

    # Output arrays
    n_preds = len(indices)
    preds = np.zeros(n_preds)

    # --- 2. Rolling Loop (Period by Period) ---
    # print(f"Running Pooled HAR-X Backtest (Window={train_win_periods} periods)...")

    for i, t_idx in enumerate(indices):

        # A. Predict current step t
        x_target_raw = X[t_idx] # (n_features, )

        # Scale X using current rolling stats
        x_scl = (x_target_raw - mean_x) / std_x

        # Predict
        pred = model.predict(x_scl.reshape(1, -1))

        # Store (Clip to prevent explosions, though rare with Log target)
        preds[i] = pred.item()

        # B. Update Model with realized value at t (Walk-Forward)
        y_realized = y[t_idx]

        # Pop Oldest
        x_old = hist_X.pop(0)
        y_old = hist_y.pop(0)

        # Update Scaler (X only)
        if use_scaling:
            scaler_x.update(x_target_raw, x_old)
            mean_x, std_x = scaler_x.get_scaler()

        # Push New Data to Buffer
        # Scale the new X with the NEW stats
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
    # Pass HAR-X settings
    hparams = {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
        # Default HAR Lags are handled inside load_and_prep_data_strided
    }

    print("Loading data...")
    # UPDATED UNPACKING: dates, baselines instead of t, adj
    X_np, y_np, dates, baselines = load_and_prep_data_strided(hparams, "all30min.parquet")

    print(f"Data Shape: X={X_np.shape}, Y={y_np.shape}")

    # Convert Day Windows to Period Windows
    periods_per_day = 48
    train_win_periods = args.train_window * periods_per_day

    # Calculate Indices
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

    if chunk_idxs.size == 0: 
        print("Chunk empty.")
        return

    # Absolute Indices for the backtester
    abs_idxs = chunk_idxs 

    # Run Pooled Backtest
    preds = run_backtest_pooled_rolling(
        abs_idxs,
        X_np, 
        y_np, 
        train_win_periods
    )

    # Save Results
    dummy_naive = np.zeros_like(preds)

    save_chunk_results(
        args.output_file, 
        preds, 
        dummy_naive, 
        chunk_idxs, 
        train_win_periods, 
        y_np, 
        dates, 
        baselines
    )
    print("Results saved.")

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