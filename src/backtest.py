import numpy as np
from tqdm import tqdm

def run_backtest_agnostic(model, indices, X, y, train_win_periods):
    """
    A truly model-agnostic walk-forward backtester.
    """
    first_test_idx = indices[0]
    if first_test_idx < train_win_periods:
        raise ValueError("Not enough history for the requested training window.")

    # 1. Provide the initial burn-in history to the model
    start_hist = first_test_idx - train_win_periods
    X_init = X[start_hist : first_test_idx]
    y_init = y[start_hist : first_test_idx]
    
    # Model handles its own scaling, buffering, and initial fitting
    model.initialize(X_init, y_init)

    n_preds = len(indices)
    preds = np.zeros(n_preds)

    # 2. Walk-Forward Loop
    for i, t_idx in enumerate(tqdm(indices, desc="Backtesting")):
        x_target = X[t_idx]
        
        # A. Predict step t
        preds[i] = model.predict(x_target)
        
        # B. Observe realized y at step t and let the model update itself
        y_realized = y[t_idx]
        model.update(x_target, y_realized)

    return preds