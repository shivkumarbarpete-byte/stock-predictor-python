"""
ml_service/train_model.py
--------------------------
Trains and compares THREE models on the same time-ordered 80/20 split:
  1. Naive Baseline  — predicts "tomorrow = today's close" (no ML at all)
  2. Linear Regression
  3. Random Forest   (100 trees, random_state=42 for reproducibility)

Why a naive baseline?
  Before showing off your ML model, you must prove it is BETTER than the
  simplest possible guess. If your ML model barely beats the naive baseline,
  it has not learned anything useful.

Why three metrics (RMSE, MAE, R²)?
  RMSE  — penalises large errors more. Good when one big wrong prediction is
           costly (e.g. in finance).
  MAE   — average error in rupees. Easy to explain to anyone.
  R²    — percentage of price variation the model explains. 1.0 = perfect.

Output files (all inside ml_service/):
  linear_regression_model_v2.pkl   — new LR model trained on 5 features (with RSI)
  random_forest_model.pkl           — RF model trained on 5 features (with RSI)
  metrics.json                      — all metrics + feature importances (read by the app)
  comparison_chart.png              — matplotlib plot of all 3 models vs actual

The ORIGINAL linear_regression_model.pkl is NEVER touched.

Run with: python ml_service/train_model.py
"""

import json
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the shared FEATURES list and file paths from config.py
# Try/except handles: Streamlit (needs "ml_service.config") vs direct run (needs "config")
try:
    from ml_service.config import FEATURES, ML_DIR, METRICS_PATH, MODEL_LR_PATH, MODEL_RF_PATH
    from ml_service.compute_indicators import add_moving_averages, add_rsi
except ImportError:
    from config import FEATURES, ML_DIR, METRICS_PATH, MODEL_LR_PATH, MODEL_RF_PATH
    from compute_indicators import add_moving_averages, add_rsi

# The target column we want to predict
TARGET = "Target_Close"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Load and prepare data
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(csv_path: Path) -> pd.DataFrame:
    """
    Reads the CSV, adds all indicators, creates the Target column, drops NaN rows.

    The Target column = next day's Close, created by shift(-1).
    The last row has no "next day" so it is also dropped.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Add all indicator columns using the same functions stock_utils.py uses
    df = add_moving_averages(df)
    df = add_rsi(df, period=14)

    # Drop rows where any feature is NaN (first ~50 rows from SMA50 warm-up)
    df = df.dropna(subset=FEATURES)

    # Create the target: what was tomorrow's closing price?
    df[TARGET] = df["Close"].shift(-1)

    # The very last row has no "next day" (NaN target) — drop it
    df = df.dropna(subset=[TARGET])

    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Time-ordered train / test split
# ──────────────────────────────────────────────────────────────────────────────

def split_by_time(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Splits data into train (oldest 80%) and test (newest 20%) sets IN TIME ORDER.
    We NEVER shuffle because shuffling creates data leakage — the model would
    accidentally see future prices during training.
    """
    split_idx = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    print(f"  Train: {len(train_df)} rows  (up to {train_df['Date'].max().date()})")
    print(f"  Test : {len(test_df)} rows  ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    return train_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Compute metrics helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates RMSE, MAE, and R² between actual and predicted values.
    Returns a dictionary so results are easy to compare and save to JSON.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    return {
        "rmse": round(rmse, 4),
        "mae":  round(mae, 4),
        "r2":   round(r2, 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Save a comparison chart
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_chart(test_df, y_test, naive_pred, lr_pred, rf_pred):
    """
    Saves a matplotlib line chart comparing all three model predictions
    against the actual prices on the test set.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(test_df["Date"], y_test,      label="Actual",           linewidth=2, color="black")
    plt.plot(test_df["Date"], naive_pred,  label="Naive Baseline",   linewidth=1.5, linestyle=":", color="gray")
    plt.plot(test_df["Date"], lr_pred,     label="Linear Regression",linewidth=1.5, linestyle="--", color="blue")
    plt.plot(test_df["Date"], rf_pred,     label="Random Forest",    linewidth=1.5, linestyle="--", color="green")

    plt.title("Model Comparison — Actual vs Predicted Close Price (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.tight_layout()

    chart_path = ML_DIR / "comparison_chart.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"  Chart saved: {chart_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — runs when you execute: python ml_service/train_model.py
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    INPUT_CSV = ML_DIR / "RELIANCE_data_with_indicators.csv"

    # If the CSV with indicators doesn't exist, fall back to raw CSV
    if not INPUT_CSV.exists():
        INPUT_CSV = ML_DIR / "RELIANCE_data.csv"
        print(f"Using raw CSV: {INPUT_CSV}")
    else:
        print(f"Using CSV with indicators: {INPUT_CSV}")

    # ── Load & prepare ──────────────────────────────────────────────────────
    print("\n[1] Preparing data ...")
    df = prepare_data(INPUT_CSV)
    print(f"  Usable rows after cleaning and dropping NaN: {len(df)}")
    print(f"  Features used: {FEATURES}")

    # ── Split ───────────────────────────────────────────────────────────────
    print("\n[2] Splitting data (80% train / 20% test, time-ordered) ...")
    train_df, test_df = split_by_time(df, test_ratio=0.2)

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df[TARGET].values

    # ── Model 1: Naive Baseline ─────────────────────────────────────────────
    # No model at all — just guess that tomorrow's price = today's Close.
    print("\n[3] Naive Baseline ...")
    naive_pred = test_df["Close"].values   # today's close = tomorrow's prediction
    naive_metrics = compute_metrics(y_test, naive_pred)
    print(f"  RMSE={naive_metrics['rmse']:.2f}  MAE={naive_metrics['mae']:.2f}  R²={naive_metrics['r2']:.4f}")

    # ── Model 2: Linear Regression ──────────────────────────────────────────
    print("\n[4] Training Linear Regression ...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred    = lr_model.predict(X_test)
    lr_metrics = compute_metrics(y_test, lr_pred)
    print(f"  RMSE={lr_metrics['rmse']:.2f}  MAE={lr_metrics['mae']:.2f}  R²={lr_metrics['r2']:.4f}")

    # ── Model 3: Random Forest ───────────────────────────────────────────────
    # n_estimators=100 means 100 decision trees are trained and averaged.
    # random_state=42 makes results reproducible (same output every run).
    print("\n[5] Training Random Forest (100 trees) ...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred    = rf_model.predict(X_test)
    rf_metrics = compute_metrics(y_test, rf_pred)
    print(f"  RMSE={rf_metrics['rmse']:.2f}  MAE={rf_metrics['mae']:.2f}  R²={rf_metrics['r2']:.4f}")

    # ── Comparison table ─────────────────────────────────────────────────────
    print("\n┌──────────────────────┬──────────┬──────────┬────────┐")
    print("│ Model                │   RMSE   │   MAE    │   R²   │")
    print("├──────────────────────┼──────────┼──────────┼────────┤")
    print(f"│ Naive Baseline       │ {naive_metrics['rmse']:8.2f} │ {naive_metrics['mae']:8.2f} │ {naive_metrics['r2']:6.4f} │")
    print(f"│ Linear Regression    │ {lr_metrics['rmse']:8.2f} │ {lr_metrics['mae']:8.2f} │ {lr_metrics['r2']:6.4f} │")
    print(f"│ Random Forest        │ {rf_metrics['rmse']:8.2f} │ {rf_metrics['mae']:8.2f} │ {rf_metrics['r2']:6.4f} │")
    print("└──────────────────────┴──────────┴──────────┴────────┘")

    # ── Feature importances (Random Forest only) ─────────────────────────────
    # Random Forest gives a score for each feature: how much does it help predictions?
    # Higher = more important. All scores add up to 1.0.
    feature_importances = {
        feature: round(float(importance), 6)
        for feature, importance in zip(FEATURES, rf_model.feature_importances_)
    }
    print("\n  Random Forest feature importances:")
    for feat, imp in sorted(feature_importances.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.4f}")

    # ── Save models to disk ──────────────────────────────────────────────────
    print("\n[6] Saving models ...")
    joblib.dump(lr_model, MODEL_LR_PATH)
    print(f"  Saved LR  -> {MODEL_LR_PATH}")

    joblib.dump(rf_model, MODEL_RF_PATH)
    print(f"  Saved RF  -> {MODEL_RF_PATH}")

    # ── Save metrics to JSON (app reads this — no need to retrain to see results) ──
    print("\n[7] Saving metrics.json ...")
    metrics_data = {
        "trained_on":    "RELIANCE.NS",
        "train_date":    str(date.today()),
        "features":      FEATURES,
        "train_rows":    int(len(train_df)),
        "test_rows":     int(len(test_df)),
        "models": [
            {"name": "Naive Baseline",    "key": "naive", **naive_metrics},
            {"name": "Linear Regression", "key": "lr",    **lr_metrics},
            {"name": "Random Forest",     "key": "rf",    **rf_metrics},
        ],
        "feature_importances": feature_importances,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Saved metrics -> {METRICS_PATH}")

    # ── Save comparison chart ────────────────────────────────────────────────
    print("\n[8] Saving comparison chart ...")
    save_comparison_chart(test_df, y_test, naive_pred, lr_pred, rf_pred)

    print("\n✅ Training complete. You can now run the Streamlit app.")
