import argparse
import os
from src.data_helper import get_chunk_indices_strided, save_chunk_results
from src.models import (
    RidgeModel,
    NaiveBaseline,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    SARIMAXModel,
)
from src.backtest import run_backtest_agnostic

def get_common_parser(description):
    """Returns a standardized arg parser for all scripts."""
    parser = argparse.ArgumentParser(description=description)
    # Added 'lightgbm' and 'random_forest' to the choices
    parser.add_argument(
        '--model',
        type=str,
        choices=['ridge', 'naive', 'xgboost', 'lightgbm', 'random_forest', 'sarimax'],
        required=True
    )
    parser.add_argument('--input-path', type=str, default="all30min")
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)
    parser.add_argument('--train-window', type=int, default=500, help="Training window in DAYS")
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")
    parser.add_argument('--lag-scope', type=str, choices=['global', 'intra'], default='global', help="Whether to compute HAR lags on the full dataset ('global') or per-segment ('intra')")
    return parser

def get_common_hparams(args):
    """Dynamically sets hparams based on the model chosen."""
    tree_models = ['xgboost', 'lightgbm']

    # Tree models usually don't need variables transformed
    use_transform = False if args.model in tree_models else True

    # XGBoost and LightGBM handle NaNs natively; other models do not.
    allow_missing = True if args.model in ['xgboost', 'lightgbm'] else False
    
    return {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
        "use_transform": use_transform,
        "allow_missing": allow_missing,
        'lag_scope': args.lag_scope
    }
    
def execute_chunk_backtest(args, hparams, X_np, y_np, dates, baselines, train_win_periods, output_file):
    """Handles model init, backtest execution, and result saving without naive baselines."""
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

    if chunk_idxs.size == 0: 
        return False 

    # 1. Initialize Model
    if args.model == 'ridge':
        print(f"  Initializing Ridge Model (Train Window: {train_win_periods} periods)...")
        model = RidgeModel(train_win_periods=train_win_periods, n_features=X_np.shape[1], use_scaling=True, alpha=1.0)
        
    elif args.model == 'naive':
        print(f"  Initializing Naive Baseline...")
        model = NaiveBaseline(lag_index=args.naive_lag)
        
    elif args.model == 'xgboost':
        print(f"  Initializing XGBoost Model (Train Window: {train_win_periods} periods)...")
        model = XGBoostModel(train_win_periods=train_win_periods, n_features=X_np.shape[1], use_scaling=False, n_estimators=100, max_depth=3, learning_rate=0.1, tree_method='hist')
        
    elif args.model == 'lightgbm':
        print(f"  Initializing LightGBM Model (Train Window: {train_win_periods} periods)...")
        model = LightGBMModel(train_win_periods=train_win_periods, n_features=X_np.shape[1], use_scaling=False, n_estimators=100, max_depth=3, learning_rate=0.1)
        
    elif args.model == 'random_forest':
        print(f"  Initializing Random Forest Model (Train Window: {train_win_periods} periods)...")
        # Note: RF doesn't use learning_rate, and standard n_estimators is often higher, but we'll keep it at 100 for speed parity
        model = RandomForestModel(train_win_periods=train_win_periods, n_features=X_np.shape[1], use_scaling=False, n_estimators=100, max_depth=3)

    elif args.model == 'sarimax':
        print(f"  Initializing SARIMAX Model (fit_window: 480 periods, refit every 48 steps)...")
        # SARIMAX(2,0,1)(1,0,0,48): ARMA(2,1) plus daily seasonal AR on 30-min bars.
        # Internally uses only the most recent 480 observations (10 trading days)
        # regardless of train_win_periods, and refits once per simulated day.
        model = SARIMAXModel(
            train_win_periods=train_win_periods,
            n_features=X_np.shape[1],
            order=(2, 0, 1),
            seasonal_order=(1, 0, 0, 48),
            fit_window=480,
            refit_frequency=48,
        )

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # 2. Run Backtest (The result of this is our primary forecast)
    preds = run_backtest_agnostic(model=model, indices=chunk_idxs, X=X_np, y=y_np, train_win_periods=train_win_periods)

    # 3. Save
    # Append _cb_drop suffix to the output filename when circuit-breaker rows
    # were dropped, so the state is visible in the filename during analysis.
    if hparams.get('cb_drop', False):
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_cb_drop{ext}"
    print(f"  Saving results to {output_file}...")
    save_chunk_results(
        output_file=output_file, 
        forecasts=preds, 
        indices=chunk_idxs, 
        train_window=train_win_periods, 
        y_true=y_np, 
        dates=dates, 
        baselines=baselines
    )
    return True