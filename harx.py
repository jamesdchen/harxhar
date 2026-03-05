import argparse
import numpy as np

# Assuming your modular files are in a 'src' folder (or in the same directory)
from src.data import load_and_prep_data_strided
from src.data_helper import get_chunk_indices_strided, save_chunk_results
from src.models import RidgeModel, NaiveBaseline
from src.backtest import run_backtest_agnostic

def main(args):
    np.random.seed(42)
    
    # 1. Setup Hyperparameters
    hparams = {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
    }

    # 2. Load Data
    print(f"Loading data from '{args.input_path}'...")
    X_np, y_np, dates, baselines = load_and_prep_data_strided(hparams, args.input_path)
    print(f"Data Shape: X={X_np.shape}, Y={y_np.shape}")

    # Convert Day Windows to Period Windows
    periods_per_day = 48
    train_win_periods = args.train_window * periods_per_day

    # 3. Calculate Indices for this Chunk
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

    if chunk_idxs.size == 0: 
        print(f"Chunk {args.chunk_id} is empty. Exiting.")
        return

    # 4. Initialize the Selected Model
    if args.model == 'ridge':
        print(f"Initializing Ridge Model (Train Window: {train_win_periods} periods)...")
        model = RidgeModel(
            train_win_periods=train_win_periods,
            n_features=X_np.shape[1],
            use_scaling=True,
            alpha=1.0  # You can expose this as an argparse argument later if needed
        )
    elif args.model == 'naive':
        print(f"Initializing Naive Baseline (Lag Index: {args.naive_lag})...")
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

    # 5. Run the Agnostic Backtest
    print("Starting backtest loop...")
    preds = run_backtest_agnostic(
        model=model, 
        indices=chunk_idxs, 
        X=X_np, 
        y=y_np, 
        train_win_periods=train_win_periods
    )

    # 6. Extract the Naive Baseline array for saving
    # We extract this directly from the features matrix so save_chunk_results 
    # can calculate the naive variance and apply the Smearing Estimator.
    naive_preds = X_np[chunk_idxs, args.naive_lag]

    # 7. Save Results
    print(f"Saving results to {args.output_file}...")
    save_chunk_results(
        output_file=args.output_file, 
        forecasts=preds, 
        naive=naive_preds, 
        indices=chunk_idxs, 
        train_window=train_win_periods, 
        y_true=y_np, 
        dates=dates, 
        baselines=baselines
    )
    print("Run complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time-Series Volatility Forecasting Pipeline")
    
    # Execution Routing
    parser.add_argument('--model', type=str, choices=['ridge', 'naive', 'xgboost'], required=True, 
                        help="Which model to run for this experiment.")
    parser.add_argument('--input-path', type=str, default="all30min", 
                        help="Path to the parquet directory or file.")
    parser.add_argument('--output-file', type=str, required=True, 
                        help="Path to save the resulting CSV.")
    
    # Chunking Options
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)

    # Model Parameters
    parser.add_argument('--train-window', type=int, default=500, 
                        help="Training window size in DAYS.")
    parser.add_argument('--exog-cols', type=str, default=None, 
                        help="Pipe-separated list of exogenous columns.")
    parser.add_argument('--naive-lag', type=int, default=0, 
                        help="Feature index to use for naive baseline (0 = 1-period MA, 1 = 5-period MA, etc.)")
    
    args = parser.parse_args()
    main(args)