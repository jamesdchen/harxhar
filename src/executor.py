import argparse
import os
from src.data_helper import get_chunk_indices_strided, save_chunk_results
from src.models import RidgeModel, NaiveBaseline, XGBoostModel
from src.backtest import run_backtest_agnostic

def get_common_parser(description):
    """Returns a standardized arg parser for all scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, choices=['ridge', 'naive', 'xgboost'], required=True)
    parser.add_argument('--input-path', type=str, default="all30min")
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)
    parser.add_argument('--train-window', type=int, default=500, help="Training window in DAYS")
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")   
    parser.add_argument('--naive-lag', type=int, default=0, help="Feature index for naive baseline")
    return parser

def get_common_hparams(args):
    """Dynamically sets hparams based on the model chosen."""
    use_log_transform = False if args.model == 'xgboost' else True
    allow_missing = True if args.model == 'xgboost' else False
    
    return {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
        "use_log": use_log_transform,
        "allow_missing": allow_missing
    }

def execute_chunk_backtest(args, hparams, X_np, y_np, dates, baselines, train_win_periods, output_file):
    """Handles model init, backtest execution, and result saving."""
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

    if chunk_idxs.size == 0: 
        return False # Indicates chunk was empty

    # Initialize Model
    if args.model == 'ridge':
        print(f"  Initializing Ridge Model (Train Window: {train_win_periods} periods)...")
        model = RidgeModel(train_win_periods=train_win_periods, n_features=X_np.shape[1], use_scaling=True, alpha=1.0)
    elif args.model == 'naive':
        print(f"  Initializing Naive Baseline (Lag: {args.naive_lag})...")
        model = NaiveBaseline(lag_index=args.naive_lag)
    elif args.model == 'xgboost':
        print(f"  Initializing XGBoost Model (Train Window: {train_win_periods} periods)...")
        model = XGBoostModel(train_win_periods=train_win_periods, n_features=X_np.shape[1], use_scaling=False, n_estimators=100, max_depth=3, learning_rate=0.1, tree_method='hist')
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Run Backtest
    preds = run_backtest_agnostic(model=model, indices=chunk_idxs, X=X_np, y=y_np, train_win_periods=train_win_periods)

    # Save
    naive_preds = X_np[chunk_idxs, args.naive_lag]
    print(f"  Saving results to {output_file}...")
    save_chunk_results(
        output_file=output_file, 
        forecasts=preds, 
        naive=naive_preds, 
        indices=chunk_idxs, 
        train_window=train_win_periods, 
        y_true=y_np, 
        dates=dates, 
        baselines=baselines,
        use_log=hparams['use_log']
    )
    return True