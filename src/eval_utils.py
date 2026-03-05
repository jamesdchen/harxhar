import os
import glob
import pandas as pd

def load_all_chunks(exp_dir, ignore_suffixes=None, require_suffixes=None):
    """
    Stitches chunk CSVs into a DataFrame with flexible filtering.
    """
    search_pattern = os.path.join(exp_dir, "results_chunk_*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        return pd.DataFrame()
    
    dfs = []
    for filename in all_files:
        base_name = os.path.splitext(os.path.basename(filename))[0] 
        
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
        except Exception as e:
            print(f"  [Warning] Could not read {base_name}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs).sort_index()

def parse_config(exp_dir):
    """Parses the config.txt file to extract the experiment name and ID."""
    config_path = os.path.join(exp_dir, "config.txt")
    exp_name = "Unknown"
    exp_id = -1
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                for line in f:
                    if line.startswith("Experiment Name:"):
                        exp_name = line.split(":", 1)[1].strip()
                    elif line.startswith("Experiment ID:"):
                        exp_id = int(line.split(":", 1)[1].strip())
        except Exception:
            pass
            
    if exp_id == -1:
        try:
            exp_id = int(exp_dir.split('_')[-1])
        except ValueError:
            pass
            
    return exp_id, exp_name

def filter_by_date(df, start_date=None, end_date=None):
    """Slices the DataFrame to the specified datetime window."""
    if df.empty or (not start_date and not end_date):
        return df
        
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
    except Exception as e:
        print(f"  [Warning] Date filtering failed: {e}")
        
    return df