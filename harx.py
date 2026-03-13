import numpy as np
from src.data import load_and_prep_data_strided
from src.executor import get_common_parser, get_common_hparams, execute_chunk_backtest

def main(args):
    np.random.seed(42)
    hparams = get_common_hparams(args)

    print(f"Loading data from '{args.input_path}'...")
    print(f"Log Transform Enabled: {hparams['use_transform']}")
    
    X_np, y_np, dates, baselines, feature_names = load_and_prep_data_strided(hparams, args.input_path)
    
    if len(X_np) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    if args.model == 'naive':
        # Feature name for the 125-period lag differs by feature_type:
        # 'har' mode → 'har_ma_125';  'raw' mode → 'adj_RV_lag_125'
        lag_key = next(f for f in feature_names if 'lag_125' in f or f == 'har_ma_125')
        args.naive_lag = feature_names.index(lag_key)

    # Static Calculation for Global Array
    periods_per_day = 48
    train_win_periods = args.train_window * periods_per_day

    # Fire off to the executor
    success = execute_chunk_backtest(
        args, hparams, X_np, y_np, dates, baselines, train_win_periods, args.output_file
    )
    
    if not success:
        print(f"Chunk {args.chunk_id} is empty. Exiting.")
    else:
        print("Run complete!")

if __name__ == '__main__':
    parser = get_common_parser("Time-Series Volatility Forecasting Pipeline")
    main(parser.parse_args())