import argparse
import os
import numpy as np
from src.backtest import get_chunk_indices_strided, save_chunk_results, run_backtest_agnostic
from src.models import create_model
from src.features import PCATransform, AETransform
from src.data import load_and_prep_data_strided, apply_horizon_shift

def add_feature_args(parser):
    """Add feature-related arguments shared between executor and submit parsers."""
    parser.add_argument('--train-window', type=int, default=500,
                        help="Training window in days")
    parser.add_argument('--n-components', type=int, default=5,
                        help="Number of PCA/AE latent components (--features pca or ae)")
    parser.add_argument('--ae-alpha', type=float, default=0.5,
                        help="AE loss weight: alpha*recon + (1-alpha)*pred (--features ae)")
    parser.add_argument('--ae-epochs', type=int, default=50,
                        help="Training epochs per AE refit (--features ae)")
    parser.add_argument('--ae-hidden', type=int, default=0,
                        help="AE hidden layer width; 0 = auto (n_features // 2) (--features ae)")
    parser.add_argument('--ae-weights-path', type=str, default=None,
                        help="Path to pre-trained AE weights .pt file (--features ae)")
    return parser


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
    add_feature_args(parser)
    parser.add_argument('--ae-loss-path', type=str, default=None,
                        help="File path to save AE training loss log CSV (--features ae)")
    parser.add_argument('--input-path', type=str, default="all30min")
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)
    parser.add_argument('--exog-cols', type=str, default=None, help="Pipe-separated list of columns")
    parser.add_argument('--lag-scope', type=str, choices=['global', 'intra'], default='global',
                        help="Whether to compute HAR lags on the full dataset ('global') or per-segment ('intra')")
    parser.add_argument('--segment', type=str, default=None,
                        choices=['all', 'morning', 'midday', 'closing', 'overnight'],
                        help="Run segmented backtest. 'all' processes every segment; or pick one.")
    parser.add_argument('--save-coefs', action='store_true', default=False,
                        help="Save rolling model coefficients to a .npz file alongside results.")
    parser.add_argument('--horizon', type=int, default=1,
                        help="Final forecast horizon H. Runs backtests for h=1,2,...,H (default 1).")
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
                           feature_names=None, horizon=1) -> bool:
    """Handles model init, backtest execution, and result saving for a single horizon."""
    chunk_idxs = get_chunk_indices_strided(X_np, train_win_periods, args.chunk_id, args.total_chunks)

    if chunk_idxs.size == 0:
        return False

    n_features = X_np.shape[1]
    feature_transform = _build_feature_transform(args, n_features)
    refit_frequency = hparams.get('refit_frequency', 1)

    print(f"  Initializing {args.model} (features={args.features}, horizon={horizon}, Train Window: {train_win_periods} periods)...")
    model = create_model(
        model_name=args.model,
        train_win_periods=train_win_periods,
        n_features=n_features,
        feature_transform=feature_transform,
        refit_frequency=refit_frequency,
        naive_lag_index=getattr(args, 'naive_lag', None),
        horizon=horizon,
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
        baselines=baselines,
        horizon=horizon,
    )

    # Save coefficients if collected
    if coef_history is not None:
        base, _ = os.path.splitext(output_file)
        coef_file = f"{base}_coefs.npz"
        chunk_dates = dates[chunk_idxs].values if hasattr(dates, 'values') else dates[chunk_idxs]
        save_kwargs = dict(coefficients=coef_history, dates=chunk_dates)
        if feature_names is not None:
            save_kwargs['feature_names'] = np.array(feature_names)
        np.savez_compressed(coef_file, **save_kwargs)
        print(f"  Saved coefficients to {coef_file}")

    return True


def main(args):
    np.random.seed(42)
    hparams = get_common_hparams(args)

    print(f"Loading data from '{args.input_path}'...")
    print(f"Tree Model: {hparams['is_tree']}")

    if args.segment is not None:
        _run_segmented(args, hparams)
    else:
        _run_global(args, hparams)


def _run_global(args, hparams):
    X_np, y_np, dates, baselines, feature_names = load_and_prep_data_strided(hparams, args.input_path)

    if len(X_np) == 0:
        print("Dataset is empty. Exiting.")
        return

    from src import config as cfg

    if args.model == 'naive':
        args.naive_lag = cfg.find_naive_lag(feature_names)

    periods_per_day = cfg.PERIODS_PER_DAY
    train_win_periods = args.train_window * periods_per_day
    final_horizon = getattr(args, 'horizon', 1)

    for h in range(1, final_horizon + 1):
        print(f"\n--- Horizon {h}/{final_horizon} ---")
        X_h, y_h, dates_h, baselines_h = apply_horizon_shift(
            X_np, y_np, dates, baselines, h)

        # Build output filename with horizon suffix
        base, ext = os.path.splitext(args.output_file)
        h_output = f"{base}_h{h}{ext}" if final_horizon > 1 else args.output_file

        success = execute_chunk_backtest(
            args, hparams, X_h, y_h, dates_h, baselines_h,
            train_win_periods, h_output,
            feature_names=feature_names, horizon=h,
        )

        if not success:
            print(f"  Chunk {args.chunk_id} is empty for horizon {h}. Skipping.")

    print("Run complete!")


def _run_segmented(args, hparams):
    datasets = load_and_prep_data_strided(hparams, args.input_path, target_segment=args.segment)

    if args.segment != 'all':
        # Single segment returned as tuple
        X_np, y_np, dates, baselines = datasets
        if len(X_np) == 0:
            print(f"No data for segment '{args.segment}'. Exiting.")
            return
        datasets = {args.segment: {'X': X_np, 'y': y_np, 'dates': dates, 'baselines': baselines}}
    elif not datasets:
        print("No datasets returned. Check data path and dates.")
        return

    if args.model == 'naive':
        from src import config as cfg
        first_ds = next(iter(datasets.values()))
        fnames = first_ds.get('features', [])
        if fnames:
            args.naive_lag = cfg.find_naive_lag(fnames)

    final_horizon = getattr(args, 'horizon', 1)

    for seg_name, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"PROCESSING SEGMENT: {seg_name.upper()}")
        print("="*50)

        dates = data['dates'] if isinstance(data, dict) else data[2]
        X = data['X'] if isinstance(data, dict) else data[0]
        y = data['y'] if isinstance(data, dict) else data[1]
        baselines = data['baselines'] if isinstance(data, dict) else data[3]

        daily_counts = dates.dt.date.value_counts()
        median_slots = int(daily_counts.median())
        train_win_periods = args.train_window * median_slots

        print(f"  Window size: {train_win_periods} rows ({args.train_window} days @ {median_slots} slots/day)")

        for h in range(1, final_horizon + 1):
            print(f"\n  --- Horizon {h}/{final_horizon} ---")
            X_h, y_h, dates_h, baselines_h = apply_horizon_shift(
                X, y, dates, baselines, h)

            base, ext = os.path.splitext(args.output_file)
            seg_output_file = f"{base}_{seg_name}_h{h}{ext}" if final_horizon > 1 else f"{base}_{seg_name}{ext}"

            seg_features = data.get('features') if isinstance(data, dict) else None
            success = execute_chunk_backtest(
                args, hparams, X_h, y_h, dates_h, baselines_h,
                train_win_periods, seg_output_file,
                feature_names=seg_features, horizon=h,
            )

            if not success:
                print(f"  [Skipping] Chunk {args.chunk_id} empty for segment {seg_name}, horizon {h}.")

    print("\nAll segments processed.")


if __name__ == '__main__':
    parser = get_common_parser("Time-Series Volatility Forecasting Pipeline")
    main(parser.parse_args())
