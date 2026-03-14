import os
import numpy as np
from src.data import load_and_prep_data_strided
from src.executor import get_common_parser, get_common_hparams, execute_chunk_backtest


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

    if args.model == 'naive':
        lag_key = next(f for f in feature_names if 'lag_125' in f or f == 'har_ma_125')
        args.naive_lag = feature_names.index(lag_key)

    from src import config as cfg
    periods_per_day = cfg.PERIODS_PER_DAY
    train_win_periods = args.train_window * periods_per_day

    success = execute_chunk_backtest(
        args, hparams, X_np, y_np, dates, baselines, train_win_periods, args.output_file,
        feature_names=feature_names,
    )

    if not success:
        print(f"Chunk {args.chunk_id} is empty. Exiting.")
    else:
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
        first_ds = next(iter(datasets.values()))
        fnames = first_ds.get('features', [])
        if fnames:
            lag_key = next(f for f in fnames if 'lag_125' in f or f == 'har_ma_125')
            args.naive_lag = fnames.index(lag_key)

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

        base, ext = os.path.splitext(args.output_file)
        seg_output_file = f"{base}_{seg_name}{ext}"

        seg_features = data.get('features') if isinstance(data, dict) else None
        success = execute_chunk_backtest(
            args, hparams, X, y, dates, baselines, train_win_periods, seg_output_file,
            feature_names=seg_features,
        )

        if not success:
            print(f"  [Skipping] Chunk {args.chunk_id} empty for segment {seg_name}.")

    print("\nAll segments processed.")


if __name__ == '__main__':
    parser = get_common_parser("Time-Series Volatility Forecasting Pipeline")
    main(parser.parse_args())
