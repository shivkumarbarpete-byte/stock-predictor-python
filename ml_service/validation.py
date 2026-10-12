"""
ml_service/validation.py
-------------------------
Walk-forward validation (expanding window, 5 folds, time-ordered, no shuffling)
comparing Naive Baseline, Linear Regression, and Random Forest.

Metrics evaluated per fold:
  - Return MAE (Mean Absolute Error on predicted vs actual next-day return)
  - Directional Accuracy (% of days the model predicted correct Up/Down direction)

Cached with st.cache_data so the Streamlit page stays fast after the first run.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    def cache_decorator(func):
        return func

try:
    from ml_service.config import FEATURES, ML_DIR
    from ml_service.train_model import prepare_data
except ImportError:
    from config import FEATURES, ML_DIR
    from train_model import prepare_data


@cache_decorator
def run_walk_forward_validation(n_splits: int = 5) -> pd.DataFrame:
    """
    Executes walk-forward expanding window validation over the dataset.
    
    Parameters:
        n_splits: Number of time-series splits (default 5)
        
    Returns:
        pd.DataFrame: Table comparing Naive, Linear Regression, and Random Forest
                      across folds on Return MAE and Directional Accuracy.
    """
    csv_path = ML_DIR / "RELIANCE_data_with_indicators.csv"
    if not csv_path.exists():
        csv_path = ML_DIR / "RELIANCE_data.csv"
        
    df = prepare_data(csv_path)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        X_train = train_df[FEATURES].values
        y_train = train_df["Target_Return"].values
        
        X_test = test_df[FEATURES].values
        y_test = test_df["Target_Return"].values
        
        # 1. Naive baseline: predicts zero return (tomorrow = today)
        naive_mae = float(mean_absolute_error(y_test, np.zeros_like(y_test)))
        
        # 2. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_mae = float(mean_absolute_error(y_test, lr_pred))
        lr_dir = float(np.mean((lr_pred > 0) == (y_test > 0)) * 100)
        
        # 3. Random Forest (100 trees, random_state=42 for reproducibility)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mae = float(mean_absolute_error(y_test, rf_pred))
        rf_dir = float(np.mean((rf_pred > 0) == (y_test > 0)) * 100)
        
        test_dates = ""
        if "Date" in test_df.columns:
            start_d = pd.to_datetime(test_df["Date"].iloc[0]).strftime("%Y-%m-%d")
            end_d = pd.to_datetime(test_df["Date"].iloc[-1]).strftime("%Y-%m-%d")
            test_dates = f"{start_d} to {end_d}"
            
        rows.append({
            "Fold": f"Fold {fold_idx}",
            "Test Period": test_dates,
            "Train / Test Size": f"{len(train_df)} / {len(test_df)}",
            "Naive Return MAE": naive_mae,
            "LR Return MAE": lr_mae,
            "RF Return MAE": rf_mae,
            "LR Dir. Accuracy": lr_dir,
            "RF Dir. Accuracy": rf_dir,
        })
        
    df_results = pd.DataFrame(rows)
    return df_results


if __name__ == "__main__":
    results = run_walk_forward_validation()
    print("Walk-forward validation results:")
    print(results.to_string(index=False))
