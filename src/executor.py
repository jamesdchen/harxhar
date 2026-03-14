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
from src.features import PCATransform, AETransform
from src.backtest import run_backtest_agnostic

def get_common_parser(description):
    """Returns a standardized arg parser for all scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--model',
        type=str,
        choices=['ridge', 'naive', 'xgboost', 'lightgbm', 'random_forest', 'sarimax'],
        required=True
    )
    parser.add_argument(
        '--features',
        type=str,
        choices=['raw', 'har', 'pca', 'ae'],
        default='raw',
        help="Feature type: raw lags, HAR rolling means, PCA-compressed, or AE-compressed"
    )
    parser.add_argument('--n-components', type=int, default=5,
                        help="Number of PCA/AE latent components (--features pca or ae)")
    parser.add_argument('--ae-alpha', type=float, default=0.5,
                        help="AE loss weight: alpha*recon + (1-alpha)*pred (--features ae)")
    parser.add_argument('--ae-epochs', type=int, default=50,
                        help="Training epochs per AE refit (--features ae)")
    parser.add_argument('--ae-hidden', type=int, default=0,
                        help="AE hidden layer width; 0 = auto (n_features // 2) (--features ae)")
    parser.add_argument('--ae-loss-path', type=str, default=None,
                        help="File path to save AE training loss log CSV (--features ae)")
    parser.add_argument('--input-path', type=str, default="all30min")
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)
    parser.add_argument('--train-window', type=int, default=500, help="Training window in DAYS")
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")
    parser.add_argument('--lag-scope', type=str, choices=['global', 'intra'], default='global',
                        help="Whether to compute HAR lags on the full dataset ('global') or per-segment ('intra')")
    return parser

def get_common_hparams(args):
    """Dynamically sets hparams based on the model and feature choices."""
    use_transform = args.model not in ('xgboost', 'lightgbm', 'random_forest')
    allow_missing = args.model in ('xgboost', 'lightgbm')
    feature_type = 'har' if args.features == 'har' else 'raw'

    return {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
        "use_transform": use_transform,
        "allow_missing": allow_missing,
        'lag_scope': args.lag_scope,
        'feature_type': feature_type,
    }

def _build_feature_transform(args, n_features):
    """Build the feature transform from CLI args, or return None for raw/har."""
    if args.features == 'pca':
        return PCATransform(n_components=args.n_components)
    elif args.features == 'ae':
        return AETransform(
            n_features=n_features,
            n_components=args.n_components,
            alpha=args.ae_alpha,
            hidden_dim=args.ae_hidden or None,
            epochs=args.ae_epochs,
            ae_loss_path=args.ae_loss_path,
        )
    return None

def execute_chunk_backtest(args, hparams, X_np, y_np, dates, baselines, train_win_periods, output_file):
    """Handles model init, backtest execution, and result saving."""
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

    if chunk_idxs.size == 0:
        return False

    n_features = X_np.shape[1]
    feature_transform = _build_feature_transform(args, n_features)
    refit_frequency = 240 if args.features == 'ae' else 1

    if args.model == 'ridge':
        print(f"  Initializing Ridge Model (features={args.features}, Train Window: {train_win_periods} periods)...")
        model = RidgeModel(
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=True,
            feature_transform=feature_transform,
            refit_frequency=refit_frequency,
            alpha=1.0,
        )

    elif args.model == 'naive':
        print(f"  Initializing Naive Baseline...")
        model = NaiveBaseline(lag_index=args.naive_lag)

    elif args.model == 'xgboost':
        print(f"  Initializing XGBoost Model (features={args.features}, Train Window: {train_win_periods} periods)...")
        model = XGBoostModel(
            train_win_periods=train_win_periods, n_features=n_features,
            use_scaling=False, feature_transform=feature_transform,
            n_estimators=100, max_depth=3, learning_rate=0.1, tree_method='hist',
        )

    elif args.model == 'lightgbm':
        print(f"  Initializing LightGBM Model (features={args.features}, Train Window: {train_win_periods} periods)...")
        model = LightGBMModel(
            train_win_periods=train_win_periods, n_features=n_features,
            use_scaling=False, feature_transform=feature_transform,
            n_estimators=100, max_depth=3, learning_rate=0.1,
        )

    elif args.model == 'random_forest':
        print(f"  Initializing Random Forest Model (features={args.features}, Train Window: {train_win_periods} periods)...")
        model = RandomForestModel(
            train_win_periods=train_win_periods, n_features=n_features,
            use_scaling=False, feature_transform=feature_transform,
            n_estimators=100, max_depth=3,
        )

    elif args.model == 'sarimax':
        print(f"  Initializing SARIMAX Model (features={args.features}, fit_window: 480 periods, refit every 48 steps)...")
        model = SARIMAXModel(
            train_win_periods=train_win_periods,
            n_features=n_features,
            order=(2, 0, 1),
            seasonal_order=(1, 0, 0, 48),
            fit_window=480,
            refit_frequency=48,
        )

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Run Backtest
    preds = run_backtest_agnostic(model=model, indices=chunk_idxs, X=X_np, y=y_np, train_win_periods=train_win_periods)

    # Save (append _cb_drop suffix when circuit-breaker rows were dropped)
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
