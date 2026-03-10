import numpy as np
import pandas as pd
from pathlib import Path

def get_chunk_indices_strided(X_np, train_window_size, chunk_id, total_chunks):
    """Calculates indices for chunked evaluation."""
    num_samples = X_np.shape[0]
    valid_test_start = train_window_size
    if valid_test_start >= num_samples: return np.array([])
    test_indices = np.arange(valid_test_start, num_samples)
    chunk_indices_list = np.array_split(test_indices, total_chunks)
    if chunk_id >= len(chunk_indices_list): return np.array([])
    return chunk_indices_list[chunk_id]

def save_chunk_results(output_file, forecasts, indices, train_window, y_true, dates, baselines):
    """Saves predictions and reconstructs raw space values for the primary model only."""
    y_subset = y_true[indices]
    base_subset = baselines[indices]
    dates_subset = dates.iloc[indices].values if hasattr(dates, 'iloc') else dates[indices]
    
    # Reconstruct from Sqrt Space using Duan's Smearing for the model
    smear = np.mean((y_subset - forecasts) ** 2)
    pred_raw = (forecasts ** 2 + smear) * base_subset
    true_raw = (y_subset ** 2) * base_subset

    # DataFrame now only contains true vs. model predicted
    df = pd.DataFrame({
        'date': dates_subset,
        'true_adj': y_subset,
        'pred_adj': forecasts,     
        'true_raw': true_raw,
        'pred_raw': pred_raw
    })
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return dates_subset