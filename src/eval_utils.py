from pathlib import Path

import pandas as pd
import numpy as np
from src.metrics import calculate_global_metrics

def load_all_chunks(exp_dir, ignore_suffixes=None, require_suffixes=None):
    """
    Stitches chunk CSVs into a DataFrame with flexible filtering.

    Files tagged with _cb_drop (e.g. results_chunk_1_cb_drop.csv) are treated
    identically to their non-tagged counterparts for the purpose of suffix
    matching — the _cb_drop token is stripped before any ignore/require check.

    Returns
    -------
    (df, cb_drop) : (pd.DataFrame, bool)
        cb_drop is True when every loaded file carries the _cb_drop tag,
        meaning the experiment ran with circuit-breaker rows excluded.
    """
    all_files = sorted(Path(exp_dir).glob("results_chunk_*.csv"))
    
    if not all_files:
        return pd.DataFrame(), False
    
    dfs = []
    cb_drop_flags = []
    for filename in all_files:
        raw_base = filename.stem

        # Strip _cb_drop before suffix matching so it is never mistaken for a
        # segment name and never accidentally triggers ignore/require filters.
        has_cb_drop = raw_base.endswith("_cb_drop")
        base_name = raw_base[: -len("_cb_drop")] if has_cb_drop else raw_base

        # 1. Check if we should ignore this file
        if ignore_suffixes and any(base_name.endswith(f"_{seg}") for seg in ignore_suffixes):
            continue
            
        # 2. Check if we strictly require a specific suffix
        if require_suffixes and not any(base_name.endswith(f"_{seg}") for seg in require_suffixes):
            continue

        try:
            df = pd.read_csv(filename)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            dfs.append(df)
            cb_drop_flags.append(has_cb_drop)
        except Exception as e:
            print(f"  [Warning] Could not read {raw_base}: {e}")
            
    if not dfs:
        return pd.DataFrame(), False

    # cb_drop is True only when every chunk file in this experiment was tagged.
    cb_drop = len(cb_drop_flags) > 0 and all(cb_drop_flags)
    return pd.concat(dfs).sort_index(), cb_drop

def parse_config(exp_dir):
    """Parses the config.txt file to extract the experiment name, ID, and model type."""
    config_path = Path(exp_dir) / "config.txt"
    exp_name = "Unknown"
    exp_id = -1
    model_type = "Unknown"

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                for line in f:
                    if line.startswith("Experiment Name:"):
                        exp_name = line.split(":", 1)[1].strip()
                    elif line.startswith("Experiment ID:"):
                        exp_id = int(line.split(":", 1)[1].strip())
                    elif line.startswith("Model Type:"):
                        model_type = line.split(":", 1)[1].strip()
        except (ValueError, IOError) as e:
            print(f"  [Warning] Could not parse config {config_path}: {e}")

    if exp_id == -1:
        try:
            exp_id = int(exp_dir.split('_')[-1])
        except (ValueError, IndexError) as e:
            print(f"  [Warning] Could not infer exp_id from path {exp_dir}: {e}")
            
    return exp_id, exp_name, model_type

def filter_by_time(df, start_time=None, end_time=None):
    """Slices the DataFrame to the specified time-of-day window."""
    if df.empty or (start_time is None and end_time is None):
        return df
        
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        start = start_time if start_time else "00:00:00"
        end = end_time if end_time else "23:59:59"
        
        # inclusive='left' prevents double-counting the exact overlapping minute (e.g., 11:30)
        return df.between_time(start, end, inclusive='left')
    except Exception as e:
        print(f"  [Warning] Time filtering failed: {e}")
        return df

def process_single_experiment(exp_dir, metadata, segment_configs):
    """Agnostically loads data, applies optional time boundaries, and calculates metrics.

    Supports multi-horizon results: if loaded data contains a 'horizon' column,
    metrics are computed per-horizon and a cross-horizon aggregate row is added.
    """
    exp_results = []

    for seg_conf in segment_configs:
        seg_name = seg_conf['name']
        load_kwargs = seg_conf['load_kwargs']
        time_bounds = seg_conf.get('time_bounds', None)

        print(f"Processing Exp {metadata['exp_id']:<3} | {metadata['model'].upper():<8} | {metadata['experiment_name'][:16]:<16} | {seg_name:<12}...", end=" ", flush=True)

        # 1. Load Data — also returns the cb_drop flag
        df, cb_drop = load_all_chunks(exp_dir, **load_kwargs)

        if df.empty:
            print("[EMPTY]")
            continue

        # 2. Apply Time-of-Day Filter in Memory
        if time_bounds:
            df = filter_by_time(df, time_bounds['start'], time_bounds['end'])
            if df.empty:
                print("[EMPTY AFTER TOD FILTER]")
                continue

        # 3. Calculate Metrics — horizon-aware
        if 'horizon' in df.columns:
            horizons = sorted(df['horizon'].unique())
            horizon_metrics = []

            for h in horizons:
                df_h = df[df['horizon'] == h]
                m = calculate_global_metrics(df_h)
                m.update(metadata)
                m['segment'] = seg_name
                m['horizon'] = int(h)
                m['cb_drop'] = cb_drop
                horizon_metrics.append(m)
                exp_results.append(m)

            # Cross-horizon aggregate
            if len(horizon_metrics) > 1:
                agg = dict(metadata)
                agg['segment'] = seg_name
                agg['horizon'] = 'mean'
                agg['cb_drop'] = cb_drop
                agg['n_samples'] = sum(m['n_samples'] for m in horizon_metrics)
                for metric_key in ('mse', 'mae', 'qlike'):
                    vals = [m[metric_key] for m in horizon_metrics if not np.isnan(m.get(metric_key, np.nan))]
                    agg[metric_key] = np.mean(vals) if vals else np.nan
                exp_results.append(agg)

            cb_tag = " [CB_DROP]" if cb_drop else ""
            print(f"[OK] {len(horizons)} horizons | n={sum(m['n_samples'] for m in horizon_metrics)}{cb_tag}")
        else:
            m = calculate_global_metrics(df)
            m.update(metadata)
            m['segment'] = seg_name
            m['horizon'] = 1
            m['cb_drop'] = cb_drop

            cb_tag = " [CB_DROP]" if cb_drop else ""
            print(f"[OK] n={m.get('n_samples', 0):<6} | QLIKE: {m.get('qlike', np.nan):.6f} | MSE: {m.get('mse', np.nan):.4e} | MAE: {m.get('mae', np.nan):.4e}{cb_tag}")

            exp_results.append(m)

    return exp_results