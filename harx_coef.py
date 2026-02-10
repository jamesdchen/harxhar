import numpy as np
import os
from sklearn.linear_model import LinearRegression, ElasticNet
from tqdm import tqdm
import argparse

# Assuming these exist in your environment
from rolling import RollingStandardScaler, RollingBuffer
from data_coef import load_and_prep_data_strided, get_chunk_indices_strided, save_chunk_results

def run_backtest_pooled_rolling(indices, X, y, train_win_periods, use_scaling=True, 
                                model_type='ols', alpha=1.0, l1_ratio=0.5):
    """
    Runs a Pooled Rolling backtest on the flattened data and captures coefficients.
    """
    n_features = X.shape[1]
    n_targets = 1
    
    # Initialize Helpers
    scaler_x = RollingStandardScaler(n_features)
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
        mean_x, std_x = scaler_x.get_scaler()
    else:
        mean_x = np.zeros(n_features)
        std_x = np.ones(n_features)

    # Fill Buffer
    buffer.X_buffer[:] = (X_init - mean_x) / std_x
    buffer.y_buffer[:] = y_init 
    
    hist_X = list(X_init)
    hist_y = list(y_init)
    
    # --- Initialize Model ---
    if model_type == 'elasticnet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, selection='random')
    else:
        model = LinearRegression(fit_intercept=True)

    # Initial Fit
    X_tr, y_tr = buffer.get_view()
    model.fit(X_tr, y_tr.ravel())
    
    # Output arrays
    n_preds = len(indices)
    preds = np.zeros(n_preds)
    coef_history = np.zeros((n_preds, n_features))
    
    # --- 2. Rolling Loop ---
    print(f"Running Pooled {model_type.upper()} Backtest (Window={train_win_periods} periods)...")
    
    for i, t_idx in enumerate(tqdm(indices)):
        
        # A. Capture Coefficients (Model fit on history t-1, valid for predicting t)
        coef_history[i, :] = model.coef_.ravel()

        # B. Predict current step t
        x_target_raw = X[t_idx]
        x_scl = (x_target_raw - mean_x) / std_x
        pred = model.predict(x_scl.reshape(1, -1))
        preds[i] = pred.item()
        
        # C. Update Model with realized value at t
        y_realized = y[t_idx]
        
        x_old = hist_X.pop(0)
        y_old = hist_y.pop(0)
        
        if use_scaling:
            scaler_x.update(x_target_raw, x_old)
            mean_x, std_x = scaler_x.get_scaler()
            
        x_new_scl = (x_target_raw - mean_x) / std_x
        buffer.add(x_new_scl, y_realized)
        
        hist_X.append(x_target_raw)
        hist_y.append(y_realized)
        
        X_tr, y_tr = buffer.get_view()
        model.fit(X_tr, y_tr.ravel())
        
    return preds, coef_history

def main(args):
    np.random.seed(42)
    hparams = {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
    }
    
    print("Loading data...")
    data_res = load_and_prep_data_strided(hparams, "all30min.csv")
    
    if len(data_res) == 5:
        X_np, y_np, dates, baselines, feature_names = data_res
        print("Feature names loaded successfully.")
    else:
        X_np, y_np, dates, baselines = data_res
        feature_names = np.array([f"Feat_{i}" for i in range(X_np.shape[1])])
        print("Warning: Using generic feature names.")

    print(f"Data Shape: X={X_np.shape}, Y={y_np.shape}")
    
    periods_per_day = 48
    train_win_periods = args.train_window * periods_per_day
    
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)
    
    if chunk_idxs.size == 0: 
        print("Chunk empty.")
        return
    
    abs_idxs = chunk_idxs 
    
    # Run Pooled Backtest
    preds, coefs = run_backtest_pooled_rolling(
        abs_idxs,
        X_np, 
        y_np, 
        train_win_periods,
        model_type=args.model_type,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio
    )
    
    dummy_naive = np.zeros_like(preds)
    
    # Save Main Results (CSV)
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
    
    # Save Coefficients (NPZ)
    base_name = os.path.splitext(args.output_file)[0]
    coef_file = f"{base_name}_coefs.npz"
    
    print(f"Saving coefficients to {coef_file}...")
    
    # --- UPDATED: Extract Dates for this Chunk ---
    # dates is a pandas Index or array. We slice it using the same indices as the backtest.
    # .values ensures we save it as a numpy array (e.g., datetime64 or object)
    chunk_dates = dates[abs_idxs].values if hasattr(dates, 'values') else dates[abs_idxs]

    np.savez_compressed(
        coef_file, 
        coefficients=coefs, 
        indices=abs_idxs, 
        dates=chunk_dates,            # <--- ADDED DATES HERE
        feature_names=feature_names   
    )
    
    print("Results saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)
    
    parser.add_argument('--train-window', type=int, default=500, help="Training window in DAYS")
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")
    
    parser.add_argument('--model-type', type=str, default='elasticnet', choices=['ols', 'elasticnet'], help="Model to use")
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha (lambda) for ElasticNet")
    parser.add_argument('--l1-ratio', type=float, default=0.5, help="L1 Ratio for ElasticNet")
    
    args = parser.parse_args()
    main(args)