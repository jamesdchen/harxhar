from __future__ import annotations

import argparse
import os

import numpy as np
from hpc.chunking import chunk_context

from core.backtest import run_backtest_agnostic, save_chunk_results
from core.core.log import get_logger
from core.data import apply_horizon_shift, load_and_prep_data_strided
from core.features import BaseFeatureTransform, create_feature_transform
from core.features.factory import REFIT_DEFAULTS
from projects.ml.cli._feature_args import add_feature_args  # re-export
from projects.ml.models import create_model

logger = get_logger(__name__)


def get_common_parser(description: str) -> argparse.ArgumentParser:
    """Returns a standardized arg parser for all scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        type=str,
        choices=["ridge", "naive", "xgboost", "lightgbm", "random_forest", "sarimax"],
        required=True,
    )
    parser.add_argument(
        "--features",
        type=str,
        choices=["har", "pca", "ae"],
        default="har",
        help="Feature type: HAR rolling means, PCA-compressed, or AE-compressed",
    )
    add_feature_args(parser)
    parser.add_argument(
        "--ae-loss-path", type=str, default=None, help="File path to save AE training loss log CSV (--features ae)"
    )
    parser.add_argument("--input-path", type=str, default="all30min")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--exog-cols", type=str, default=None, help="Pipe-separated list of columns")
    parser.add_argument(
        "--lag-scope",
        type=str,
        choices=["global", "intra"],
        default="global",
        help="Whether to compute HAR lags on the full dataset ('global') or per-segment ('intra')",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default=None,
        choices=["all", "morning", "midday", "closing", "overnight"],
        help="Run segmented backtest. 'all' processes every segment; or pick one.",
    )
    parser.add_argument(
        "--save-coefs",
        action="store_true",
        default=False,
        help="Save rolling model coefficients to a .npz file alongside results.",
    )
    parser.add_argument(
        "--horizon", type=int, default=1, help="Final forecast horizon H. Runs backtests for h=1,2,...,H (default 1)."
    )
    return parser


def get_common_hparams(args: argparse.Namespace) -> dict[str, object]:
    """Dynamically sets hparams based on the model and feature choices."""
    is_tree = args.model in ("xgboost", "lightgbm", "random_forest")
    allow_missing = args.model in ("xgboost", "lightgbm")
    feature_type = args.features

    if args.model in ("xgboost", "lightgbm", "random_forest"):
        refit_frequency = 5
    else:
        refit_frequency = REFIT_DEFAULTS.get(args.features, 1)

    return {
        "diurnal_adjust": True,
        "exog_cols": args.exog_cols,
        "is_tree": is_tree,
        "use_transform_exog": not is_tree,
        "use_diurnal": not is_tree,
        "use_winsor": not is_tree,
        "allow_missing": allow_missing,
        "lag_scope": args.lag_scope,
        "feature_type": feature_type,
        "refit_frequency": refit_frequency,
    }


def _build_feature_transform(args: argparse.Namespace, n_features: int) -> BaseFeatureTransform | None:
    """Build the feature transform from CLI args, or return None for raw/har."""
    return create_feature_transform(
        kind=args.features,
        n_components=args.n_components,
        n_features=n_features,
        alpha=getattr(args, "ae_alpha", 0.5),
        hidden_dim=getattr(args, "ae_hidden", None) or None,
        epochs=getattr(args, "ae_epochs", 50),
        ae_loss_path=getattr(args, "ae_loss_path", None),
        ae_weights_path=getattr(args, "ae_weights_path", None),
    )


def execute_chunk_backtest(
    args,
    hparams: dict,
    X_np,
    y_np,
    dates,
    baselines,
    train_win_periods: int,
    output_file: str,
    feature_names=None,
    horizon=1,
) -> bool:
    """Handles model init, backtest execution, and result saving for a single horizon."""
    ctx = chunk_context()
    chunk_range = ctx.split(range(train_win_periods, len(X_np)))
    chunk_idxs = np.arange(chunk_range.start, chunk_range.stop)

    if chunk_idxs.size == 0:
        return False

    n_features = X_np.shape[1]
    feature_transform = _build_feature_transform(args, n_features)
    refit_frequency = hparams.get("refit_frequency", 1)

    logger.info(
        "Initializing %s (features=%s, horizon=%d, Train Window: %d periods)",
        args.model,
        args.features,
        horizon,
        train_win_periods,
    )
    model = create_model(
        model_name=args.model,
        train_win_periods=train_win_periods,
        n_features=n_features,
        feature_transform=feature_transform,
        refit_frequency=refit_frequency,
        naive_lag_index=getattr(args, "naive_lag", None),
        horizon=horizon,
    )

    # Run Backtest
    save_coefs = getattr(args, "save_coefs", False)
    preds, coef_history = run_backtest_agnostic(
        model=model,
        indices=chunk_idxs,
        X=X_np,
        y=y_np,
        train_win_periods=train_win_periods,
        save_coefs=save_coefs,
    )

    logger.info("Saving results to %s", output_file)
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
        chunk_dates = dates[chunk_idxs].values if hasattr(dates, "values") else dates[chunk_idxs]
        save_kwargs = dict(coefficients=coef_history, dates=chunk_dates)
        if feature_names is not None:
            save_kwargs["feature_names"] = np.array(feature_names)
        np.savez_compressed(coef_file, **save_kwargs)
        logger.info("Saved coefficients to %s", coef_file)

    return True


def main(args: argparse.Namespace) -> None:
    if args.output_file is None:
        args.output_file = str(chunk_context().output_path())

    hparams = get_common_hparams(args)

    logger.info("Loading data from '%s'", args.input_path)
    logger.info("Tree Model: %s", hparams["is_tree"])

    if args.segment is not None:
        _run_segmented(args, hparams)
    else:
        _run_global(args, hparams)


def _run_global(args: argparse.Namespace, hparams: dict) -> None:
    from core.core import config as cfg

    # Naive baseline uses hardcoded raw lags
    if args.model == "naive":
        hparams = {**hparams, "feature_type": "raw"}
        X_np, y_np, dates, baselines, feature_names = load_and_prep_data_strided(
            hparams, args.input_path, lag=cfg.NAIVE_LAG
        )
    else:
        X_np, y_np, dates, baselines, feature_names = load_and_prep_data_strided(hparams, args.input_path)

    if len(X_np) == 0:
        logger.info("Dataset is empty. Exiting.")
        return

    if args.model == "naive":
        args.naive_lag = cfg.find_naive_lag(feature_names)

    periods_per_day = cfg.PERIODS_PER_DAY
    train_win_periods = args.train_window * periods_per_day
    final_horizon = getattr(args, "horizon", 1)

    for h in range(1, final_horizon + 1):
        logger.info("--- Horizon %d/%d ---", h, final_horizon)
        X_h, y_h, dates_h, baselines_h = apply_horizon_shift(X_np, y_np, dates, baselines, h)

        # Build output filename with horizon suffix
        base, ext = os.path.splitext(args.output_file)
        h_output = f"{base}_h{h}{ext}" if final_horizon > 1 else args.output_file

        success = execute_chunk_backtest(
            args,
            hparams,
            X_h,
            y_h,
            dates_h,
            baselines_h,
            train_win_periods,
            h_output,
            feature_names=feature_names,
            horizon=h,
        )

        if not success:
            logger.info("Chunk is empty for horizon %d. Skipping.", h)

    logger.info("Run complete!")


def _run_segmented(args: argparse.Namespace, hparams: dict) -> None:
    from core.core import config as cfg

    # Naive baseline uses hardcoded raw lags
    if args.model == "naive":
        hparams = {**hparams, "feature_type": "raw"}
        datasets = load_and_prep_data_strided(hparams, args.input_path, target_segment=args.segment, lag=cfg.NAIVE_LAG)
    else:
        datasets = load_and_prep_data_strided(hparams, args.input_path, target_segment=args.segment)

    if args.segment != "all":
        # Single segment returned as tuple
        X_np, y_np, dates, baselines = datasets
        if len(X_np) == 0:
            logger.info("No data for segment '%s'. Exiting.", args.segment)
            return
        datasets = {args.segment: {"X": X_np, "y": y_np, "dates": dates, "baselines": baselines}}
    elif not datasets:
        logger.info("No datasets returned. Check data path and dates.")
        return

    assert isinstance(datasets, dict)

    if args.model == "naive":
        first_ds = next(iter(datasets.values()))
        fnames = first_ds.get("features", [])
        if fnames:
            args.naive_lag = cfg.find_naive_lag(fnames)

    final_horizon = getattr(args, "horizon", 1)

    for seg_name, data in datasets.items():
        logger.info("%s PROCESSING SEGMENT: %s %s", "=" * 20, seg_name.upper(), "=" * 20)

        dates = data["dates"] if isinstance(data, dict) else data[2]
        X = data["X"] if isinstance(data, dict) else data[0]
        y = data["y"] if isinstance(data, dict) else data[1]
        baselines = data["baselines"] if isinstance(data, dict) else data[3]

        daily_counts = dates.dt.date.value_counts()
        median_slots = int(daily_counts.median())
        train_win_periods = args.train_window * median_slots

        logger.info("Window size: %d rows (%d days @ %d slots/day)", train_win_periods, args.train_window, median_slots)

        for h in range(1, final_horizon + 1):
            logger.info("--- Horizon %d/%d ---", h, final_horizon)
            X_h, y_h, dates_h, baselines_h = apply_horizon_shift(X, y, dates, baselines, h)

            base, ext = os.path.splitext(args.output_file)
            seg_output_file = f"{base}_{seg_name}_h{h}{ext}" if final_horizon > 1 else f"{base}_{seg_name}{ext}"

            seg_features = data.get("features") if isinstance(data, dict) else None
            success = execute_chunk_backtest(
                args,
                hparams,
                X_h,
                y_h,
                dates_h,
                baselines_h,
                train_win_periods,
                seg_output_file,
                feature_names=seg_features,
                horizon=h,
            )

            if not success:
                logger.info("[Skipping] Chunk empty for segment %s, horizon %d.", seg_name, h)

    logger.info("All segments processed.")


if __name__ == "__main__":
    np.random.seed(42)
    parser = get_common_parser("Time-Series Volatility Forecasting Pipeline")
    main(parser.parse_args())
