"""
ml_service/train_model.py
--------------------------
Trains and compares THREE models on the same time-ordered 80/20 split:
  1. Naive Baseline  - predicts "tomorrow = today's close" (return = 0)
  2. Linear Regression
  3. Random Forest   (100 trees, random_state=42)

NEW (v3): models predict the next-day RETURN (%), not the next-day price.
  predicted_price = Close * (1 + predicted_return)
Features are scale-free ratios (see features.py), so the model is not tied
to the price level of the stock it was trained on.

Output files (all inside ml_service/):
  linear_regression_model_v3.pkl
  random_forest_model_v3.pkl
  metrics.json          - price metrics (same format as before) + extra_metrics
  comparison_chart.png

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

try:
    from ml_service.config import FEATURES, ML_DIR, METRICS_PATH, MODEL_LR_PATH, MODEL_RF_PATH
    from ml_service.compute_indicators import add_moving_averages, add_rsi
    from ml_service.features import add_scale_free_features
except ImportError:
    from config import FEATURES, ML_DIR, METRICS_PATH, MODEL_LR_PATH, MODEL_RF_PATH
    from compute_indicators import add_moving_averages, add_rsi
    from features import add_scale_free_features

# What the model learns to predict: tomorrow's % return (0.01 = +1%)
TARGET = "Target_Return"
# Tomorrow's actual Close in Rs - used only to measure errors in rupees
NEXT_CLOSE = "Target_Close"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Load and prepare data
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Safety: make sure rows are oldest -> newest, otherwise shift(-1) is wrong
    df = df.sort_values("Date").reset_index(drop=True)

    # Raw indicators (same functions stock_utils.py uses)
    df = add_moving_averages(df)
    df = add_rsi(df, period=14)

    # Scale-free features (SMA20_ratio, SMA50_ratio, EMA20_ratio)
    df = add_scale_free_features(df)

    # Drop warm-up rows where any feature is NaN
    df = df.dropna(subset=FEATURES)

    # Targets: tomorrow's Close, and tomorrow's return
    df[NEXT_CLOSE] = df["Close"].shift(-1)
    df[TARGET] = df[NEXT_CLOSE] / df["Close"] - 1

    # Last row has no "tomorrow" -> drop it
    df = df.dropna(subset=[NEXT_CLOSE, TARGET])

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Time-ordered train / test split
# ──────────────────────────────────────────────────────────────────────────────

def split_by_time(df: pd.DataFrame, test_ratio: float = 0.2):
    split_idx = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"  Train: {len(train_df)} rows  (up to {train_df['Date'].max().date()})")
    print(f"  Test : {len(test_df)} rows  ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    return train_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R² - computed on PRICES (Rs), same as before."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 6)}


def compute_return_metrics(ret_true: np.ndarray, ret_pred: np.ndarray) -> dict:
    """
    Honest metrics on RETURNS (price R² is ~0.99 for everyone, so it can't
    tell models apart).
      r2_return : > 0 means better than guessing the average return
      dir_acc   : % of days the model got UP/DOWN direction right (50% = coin flip)
    """
    r2_ret = float(r2_score(ret_true, ret_pred))
    dir_acc = float(np.mean(np.sign(ret_true) == np.sign(ret_pred)))
    return {"r2_return": round(r2_ret, 6), "dir_acc": round(dir_acc, 4)}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Comparison chart (in rupees)
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_chart(test_df, y_test, naive_pred, lr_pred, rf_pred):
    plt.figure(figsize=(14, 6))
    plt.plot(test_df["Date"], y_test, label="Actual", linewidth=2, color="black")
    plt.plot(test_df["Date"], naive_pred, label="Naive Baseline", linewidth=1.5, linestyle=":", color="gray")
    plt.plot(test_df["Date"], lr_pred, label="Linear Regression", linewidth=1.5, linestyle="--", color="blue")
    plt.plot(test_df["Date"], rf_pred, label="Random Forest", linewidth=1.5, linestyle="--", color="green")

    plt.title("Model Comparison - Actual vs Predicted Close Price (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.tight_layout()

    chart_path = ML_DIR / "comparison_chart.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"  Chart saved: {chart_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    INPUT_CSV = ML_DIR / "RELIANCE_data_with_indicators.csv"
    if not INPUT_CSV.exists():
        INPUT_CSV = ML_DIR / "RELIANCE_data.csv"
        print(f"Using raw CSV: {INPUT_CSV}")
    else:
        print(f"Using CSV with indicators: {INPUT_CSV}")

    print("\n[1] Preparing data ...")
    df = prepare_data(INPUT_CSV)
    print(f"  Usable rows after cleaning and dropping NaN: {len(df)}")
    print(f"  Features used: {FEATURES}")

    print("\n[2] Splitting data (80% train / 20% test, time-ordered) ...")
    train_df, test_df = split_by_time(df, test_ratio=0.2)

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values          # returns
    X_test = test_df[FEATURES].values
    y_test_ret = test_df[TARGET].values        # actual returns
    y_test_price = test_df[NEXT_CLOSE].values  # actual next-day prices (Rs)
    close_test = test_df["Close"].values       # today's close, used to rebuild price

    # ── Model 1: Naive Baseline (return = 0 -> price = today's Close) ───────
    print("\n[3] Naive Baseline ...")
    naive_ret_pred = np.zeros(len(test_df))
    naive_pred = close_test * (1 + naive_ret_pred)
    naive_metrics = compute_metrics(y_test_price, naive_pred)
    naive_extra = {"r2_return": round(float(r2_score(y_test_ret, naive_ret_pred)), 6), "dir_acc": None}
    print(f"  RMSE={naive_metrics['rmse']:.2f}  MAE={naive_metrics['mae']:.2f}  R²={naive_metrics['r2']:.4f}")

    # ── Model 2: Linear Regression ──────────────────────────────────────────
    print("\n[4] Training Linear Regression ...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_ret_pred = lr_model.predict(X_test)
    lr_pred = close_test * (1 + lr_ret_pred)
    lr_metrics = compute_metrics(y_test_price, lr_pred)
    lr_extra = compute_return_metrics(y_test_ret, lr_ret_pred)
    print(f"  RMSE={lr_metrics['rmse']:.2f}  MAE={lr_metrics['mae']:.2f}  R²={lr_metrics['r2']:.4f}")

    # ── Model 3: Random Forest ───────────────────────────────────────────────
    print("\n[5] Training Random Forest (100 trees) ...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_ret_pred = rf_model.predict(X_test)
    rf_pred = close_test * (1 + rf_ret_pred)
    rf_metrics = compute_metrics(y_test_price, rf_pred)
    rf_extra = compute_return_metrics(y_test_ret, rf_ret_pred)
    print(f"  RMSE={rf_metrics['rmse']:.2f}  MAE={rf_metrics['mae']:.2f}  R²={rf_metrics['r2']:.4f}")

    # ── Comparison tables ────────────────────────────────────────────────────
    print("\nPRICE metrics (Rs):")
    print("┌──────────────────────┬──────────┬──────────┬────────┐")
    print("│ Model                │   RMSE   │   MAE    │   R²   │")
    print("├──────────────────────┼──────────┼──────────┼────────┤")
    print(f"│ Naive Baseline       │ {naive_metrics['rmse']:8.2f} │ {naive_metrics['mae']:8.2f} │ {naive_metrics['r2']:6.4f} │")
    print(f"│ Linear Regression    │ {lr_metrics['rmse']:8.2f} │ {lr_metrics['mae']:8.2f} │ {lr_metrics['r2']:6.4f} │")
    print(f"│ Random Forest        │ {rf_metrics['rmse']:8.2f} │ {rf_metrics['mae']:8.2f} │ {rf_metrics['r2']:6.4f} │")
    print("└──────────────────────┴──────────┴──────────┴────────┘")

    print("\nRETURN metrics (the honest ones):")
    print(f"  Naive : R²_return={naive_extra['r2_return']:.4f}")
    print(f"  LR    : R²_return={lr_extra['r2_return']:.4f}  direction accuracy={lr_extra['dir_acc']*100:.1f}%")
    print(f"  RF    : R²_return={rf_extra['r2_return']:.4f}  direction accuracy={rf_extra['dir_acc']*100:.1f}%")

    # ── Feature importances (Random Forest) ──────────────────────────────────
    feature_importances = {
        feature: round(float(importance), 6)
        for feature, importance in zip(FEATURES, rf_model.feature_importances_)
    }
    print("\n  Random Forest feature importances:")
    for feat, imp in sorted(feature_importances.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.4f}")

    # ── Save models ──────────────────────────────────────────────────────────
    print("\n[6] Saving models ...")
    joblib.dump(lr_model, MODEL_LR_PATH)
    print(f"  Saved LR  -> {MODEL_LR_PATH}")
    joblib.dump(rf_model, MODEL_RF_PATH)
    print(f"  Saved RF  -> {MODEL_RF_PATH}")

    # ── Save metrics.json (same "models" format as before; extras separate) ──
    print("\n[7] Saving metrics.json ...")
    metrics_data = {
        "trained_on": "RELIANCE.NS",
        "train_date": str(date.today()),
        "features": FEATURES,
        "target": "next_day_return",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "models": [
            {"name": "Naive Baseline", "key": "naive", **naive_metrics},
            {"name": "Linear Regression", "key": "lr", **lr_metrics},
            {"name": "Random Forest", "key": "rf", **rf_metrics},
        ],
        "extra_metrics": {"naive": naive_extra, "lr": lr_extra, "rf": rf_extra},
        "feature_importances": feature_importances,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Saved metrics -> {METRICS_PATH}")

    print("\n[8] Saving comparison chart ...")
    save_comparison_chart(test_df, y_test_price, naive_pred, lr_pred, rf_pred)

    print("\n✅ Training complete.")