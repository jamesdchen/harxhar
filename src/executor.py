import argparse
import os
from src.backtest_helper import get_chunk_indices_strided, save_chunk_results
from src.models import create_model
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
    parser.add_argument('--ae-weights-path', type=str, default=None,
                        help="Path to pre-trained AE weights .pt file. "
                             "When set, AE skips training and uses loaded weights (--features ae)")
    parser.add_argument('--input-path', type=str, default="all30min")
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)
    parser.add_argument('--train-window', type=int, default=500, help="Training window in DAYS")
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")
    parser.add_argument('--lag-scope', type=str, choices=['global', 'intra'], default='global',
                        help="Whether to compute HAR lags on the full dataset ('global') or per-segment ('intra')")
    parser.add_argument('--segment', type=str, default=None,
                        choices=['all', 'morning', 'midday', 'closing', 'overnight'],
                        help="Run segmented backtest. 'all' processes every segment; or pick one.")
    parser.add_argument('--save-coefs', action='store_true', default=False,
                        help="Save rolling model coefficients to a .npz file alongside results.")
    return parser

def get_common_hparams(args):
    """Dynamically sets hparams based on the model and feature choices."""
    is_tree = args.model in ('xgboost', 'lightgbm', 'random_forest')
    allow_missing = args.model in ('xgboost', 'lightgbm')
    feature_type = 'har' if args.features == 'har' else 'raw'

    from src import config as cfg
    refit_frequency = cfg.AE_REFIT_FREQUENCY if args.features == 'ae' else 1

    return {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
        "is_tree": is_tree,
        "use_transform_exog": not is_tree,
        "use_diurnal": not is_tree,
        "use_winsor": not is_tree,
        "allow_missing": allow_missing,
        'lag_scope': args.lag_scope,
        'feature_type': feature_type,
        'refit_frequency': refit_frequency,
    }

def _build_feature_transform(args, n_features):
    """Build the feature transform from CLI args, or return None for raw/har."""
    if args.features == 'pca':
        return PCATransform(n_components=args.n_components)
    elif args.features == 'ae':
        transform = AETransform(
            n_features=n_features,
            n_components=args.n_components,
            alpha=args.ae_alpha,
            hidden_dim=args.ae_hidden or None,
            epochs=args.ae_epochs,
            ae_loss_path=args.ae_loss_path,
        )
        if args.ae_weights_path is not None:
            transform.load_weights(args.ae_weights_path)
        return transform
    return None

def execute_chunk_backtest(args, hparams: dict, X_np, y_np, dates, baselines, train_win_periods: int, output_file: str,
                           feature_names=None) -> bool:
    """Handles model init, backtest execution, and result saving."""
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

    if chunk_idxs.size == 0:
        return False

    n_features = X_np.shape[1]
    feature_transform = _build_feature_transform(args, n_features)
    refit_frequency = hparams.get('refit_frequency', 1)

    print(f"  Initializing {args.model} (features={args.features}, Train Window: {train_win_periods} periods)...")
    model = create_model(
        model_name=args.model,
        train_win_periods=train_win_periods,
        n_features=n_features,
        feature_transform=feature_transform,
        refit_frequency=refit_frequency,
        naive_lag_index=getattr(args, 'naive_lag', None),
    )

    # Run Backtest
    save_coefs = getattr(args, 'save_coefs', False)
    preds, coef_history = run_backtest_agnostic(
        model=model, indices=chunk_idxs, X=X_np, y=y_np,
        train_win_periods=train_win_periods, save_coefs=save_coefs,
    )

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

    # Save coefficients if collected
    if coef_history is not None:
        import numpy as np
        base, _ = os.path.splitext(output_file)
        coef_file = f"{base}_coefs.npz"
        chunk_dates = dates[chunk_idxs].values if hasattr(dates, 'values') else dates[chunk_idxs]
        save_kwargs = dict(coefficients=coef_history, dates=chunk_dates)
        if feature_names is not None:
            save_kwargs['feature_names'] = np.array(feature_names)
        np.savez_compressed(coef_file, **save_kwargs)
        print(f"  Saved coefficients to {coef_file}")

    return True
