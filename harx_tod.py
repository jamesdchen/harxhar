import os
import numpy as np
from src.data_tod import load_and_prep_data_strided
from src.executor import get_common_parser, get_common_hparams, execute_chunk_backtest

def main(args):
    np.random.seed(42)
    hparams = get_common_hparams(args)

    print(f"Loading segmented data from {args.input_path}...")
    print(f"Log Transform Enabled: {hparams['use_transform']}")
    
    datasets = load_and_prep_data_strided(hparams, args.input_path, target_segment='all')

    if not datasets:
        print("No datasets returned. Check data path and dates.")
        return

    if args.model == 'naive':
        first_ds = next(iter(datasets.values()))
        fnames = first_ds['features']
        lag_key = next(f for f in fnames if 'lag_125' in f or f == 'har_ma_125')
        args.naive_lag = fnames.index(lag_key)

    for seg_name, data in datasets.items():
        print(f"\n" + "="*50)
        print(f"PROCESSING SEGMENT: {seg_name.upper()}")
        print("="*50)

        # Dynamic Window Calculation
        dates = data['dates']
        daily_counts = dates.dt.date.value_counts()
        median_slots = int(daily_counts.median())
        train_win_periods = args.train_window * median_slots
        
        print(f"  Window size: {train_win_periods} rows ({args.train_window} days @ {median_slots} slots/day)")

        # Format output string
        base, ext = os.path.splitext(args.output_file)
        seg_output_file = f"{base}_{seg_name}{ext}"

        # Fire off to the executor
        success = execute_chunk_backtest(
            args, hparams, data['X'], data['y'], dates, data['baselines'], train_win_periods, seg_output_file
        )
        
        if not success:
            print(f"  [Skipping] Chunk {args.chunk_id} empty for segment {seg_name}.")

    print("\nAll segments processed.")

if __name__ == '__main__':
    parser = get_common_parser("Segmented Time-Series Backtester")
    main(parser.parse_args())