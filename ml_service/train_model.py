"""
ml_service/train_model.py
--------------------------
Trains and compares FIVE models on the same time-ordered 80/20 split:
  1. Naive Baseline  - predicts "tomorrow = today's close" (return = 0)
  2. Linear Regression
  3. Random Forest   (hyperparameter-tuned via RandomizedSearchCV)
  4. XGBoost         (hyperparameter-tuned via RandomizedSearchCV)
  5. LightGBM        (sensible defaults with early-stopping via eval set)

v4 additions (on top of v3):
  - XGBoost and LightGBM models
  - Automated hyperparameter tuning for Random Forest & XGBoost
  - Sentiment_Score as a feature (defaults to 0.0 during training on
    historical CSVs because we lack archived headlines)

Output files (all inside ml_service/):
  linear_regression_model_v3.pkl
  random_forest_model_v3.pkl
  xgboost_model_v3.pkl
  lightgbm_model_v3.pkl
  metrics.json          - price metrics + extra_metrics + tuning info
  comparison_chart.png

Run with: python ml_service/train_model.py
"""

import json
import warnings
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# XGBoost & LightGBM imports with graceful fallback
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("⚠️  xgboost not installed. Skipping XGBoost. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("⚠️  lightgbm not installed. Skipping LightGBM. Install with: pip install lightgbm")

try:
    from ml_service.config import (
        FEATURES, ML_DIR, METRICS_PATH,
        MODEL_LR_PATH, MODEL_RF_PATH, MODEL_XGB_PATH, MODEL_LGB_PATH,
    )
    from ml_service.compute_indicators import add_moving_averages, add_rsi
    from ml_service.features import add_scale_free_features, add_sentiment_feature
except ImportError:
    from config import (
        FEATURES, ML_DIR, METRICS_PATH,
        MODEL_LR_PATH, MODEL_RF_PATH, MODEL_XGB_PATH, MODEL_LGB_PATH,
    )
    from compute_indicators import add_moving_averages, add_rsi
    from features import add_scale_free_features, add_sentiment_feature

# Suppress sklearn convergence / parameter warnings during tuning
warnings.filterwarnings("ignore", category=UserWarning)

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

    # Sentiment feature — defaults to 0.0 for historical training data
    # because we don't have archived headlines for every trading day.
    df = add_sentiment_feature(df, sentiment_score=0.0)

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
# STEP 4: Hyperparameter tuning
# ──────────────────────────────────────────────────────────────────────────────

def tune_random_forest(X_train, y_train, n_iter: int = 30, cv_splits: int = 3):
    """
    Tunes Random Forest using RandomizedSearchCV with TimeSeriesSplit.
    Returns the best estimator and the best parameters found.
    """
    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    base_model = RandomForestRegressor(random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"  Best RF params: {search.best_params_}")
    print(f"  Best CV score (neg MSE): {search.best_score_:.6f}")

    return search.best_estimator_, search.best_params_


def tune_xgboost(X_train, y_train, n_iter: int = 30, cv_splits: int = 3):
    """
    Tunes XGBoost using RandomizedSearchCV with TimeSeriesSplit.
    Returns the best estimator and the best parameters found.
    """
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "min_child_weight": [1, 3, 5, 7],
    }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"  Best XGB params: {search.best_params_}")
    print(f"  Best CV score (neg MSE): {search.best_score_:.6f}")

    return search.best_estimator_, search.best_params_


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Comparison chart (in rupees)
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_chart(test_df, y_test, predictions: dict):
    """
    Plots actual vs predicted prices for all models.

    Args:
        test_df: test DataFrame with Date column
        y_test: actual next-day prices
        predictions: dict of {label: predicted_prices_array}
    """
    colors = {
        "Naive Baseline":     ("gray",    ":"),
        "Linear Regression":  ("blue",    "--"),
        "Random Forest":      ("green",   "--"),
        "XGBoost":            ("orange",  "-."),
        "LightGBM":           ("purple",  "-."),
    }

    plt.figure(figsize=(14, 6))
    plt.plot(test_df["Date"], y_test, label="Actual", linewidth=2, color="black")

    for label, pred_vals in predictions.items():
        color, ls = colors.get(label, ("red", "-"))
        plt.plot(test_df["Date"], pred_vals, label=label, linewidth=1.5,
                 linestyle=ls, color=color)

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

    # ── Collect all model results ────────────────────────────────────────────
    models_list = []
    extra_metrics = {}
    chart_preds = {}
    trained_models = {}
    best_params_all = {}

    # ── Model 1: Naive Baseline (return = 0 -> price = today's Close) ───────
    print("\n[3] Naive Baseline ...")
    naive_ret_pred = np.zeros(len(test_df))
    naive_pred = close_test * (1 + naive_ret_pred)
    naive_metrics = compute_metrics(y_test_price, naive_pred)
    naive_extra = {"r2_return": round(float(r2_score(y_test_ret, naive_ret_pred)), 6), "dir_acc": None}
    print(f"  RMSE={naive_metrics['rmse']:.2f}  MAE={naive_metrics['mae']:.2f}  R²={naive_metrics['r2']:.4f}")

    models_list.append({"name": "Naive Baseline", "key": "naive", **naive_metrics})
    extra_metrics["naive"] = naive_extra
    chart_preds["Naive Baseline"] = naive_pred

    # ── Model 2: Linear Regression ──────────────────────────────────────────
    print("\n[4] Training Linear Regression ...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_ret_pred = lr_model.predict(X_test)
    lr_pred = close_test * (1 + lr_ret_pred)
    lr_metrics = compute_metrics(y_test_price, lr_pred)
    lr_extra = compute_return_metrics(y_test_ret, lr_ret_pred)
    print(f"  RMSE={lr_metrics['rmse']:.2f}  MAE={lr_metrics['mae']:.2f}  R²={lr_metrics['r2']:.4f}")

    models_list.append({"name": "Linear Regression", "key": "lr", **lr_metrics})
    extra_metrics["lr"] = lr_extra
    chart_preds["Linear Regression"] = lr_pred
    trained_models["lr"] = lr_model

    # ── Model 3: Random Forest (hyperparameter-tuned) ────────────────────────
    print("\n[5] Tuning & Training Random Forest ...")
    rf_model, rf_best_params = tune_random_forest(X_train, y_train, n_iter=30, cv_splits=3)
    rf_ret_pred = rf_model.predict(X_test)
    rf_pred = close_test * (1 + rf_ret_pred)
    rf_metrics = compute_metrics(y_test_price, rf_pred)
    rf_extra = compute_return_metrics(y_test_ret, rf_ret_pred)
    print(f"  RMSE={rf_metrics['rmse']:.2f}  MAE={rf_metrics['mae']:.2f}  R²={rf_metrics['r2']:.4f}")

    models_list.append({"name": "Random Forest", "key": "rf", **rf_metrics})
    extra_metrics["rf"] = rf_extra
    chart_preds["Random Forest"] = rf_pred
    trained_models["rf"] = rf_model
    best_params_all["rf"] = {k: (v if not isinstance(v, np.integer) else int(v))
                             for k, v in rf_best_params.items()}

    # ── Model 4: XGBoost (hyperparameter-tuned) ─────────────────────────────
    if _HAS_XGB:
        print("\n[6] Tuning & Training XGBoost ...")
        xgb_model, xgb_best_params = tune_xgboost(X_train, y_train, n_iter=30, cv_splits=3)
        xgb_ret_pred = xgb_model.predict(X_test)
        xgb_pred = close_test * (1 + xgb_ret_pred)
        xgb_metrics = compute_metrics(y_test_price, xgb_pred)
        xgb_extra = compute_return_metrics(y_test_ret, xgb_ret_pred)
        print(f"  RMSE={xgb_metrics['rmse']:.2f}  MAE={xgb_metrics['mae']:.2f}  R²={xgb_metrics['r2']:.4f}")

        models_list.append({"name": "XGBoost", "key": "xgb", **xgb_metrics})
        extra_metrics["xgb"] = xgb_extra
        chart_preds["XGBoost"] = xgb_pred
        trained_models["xgb"] = xgb_model
        best_params_all["xgb"] = {k: (v if not isinstance(v, (np.integer, np.floating))
                                       else (int(v) if isinstance(v, np.integer) else float(v)))
                                  for k, v in xgb_best_params.items()}
    else:
        print("\n[6] Skipping XGBoost (not installed)")

    # ── Model 5: LightGBM ───────────────────────────────────────────────────
    if _HAS_LGB:
        print("\n[7] Training LightGBM ...")
        lgb_model = LGBMRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=5,
            random_state=42,
            verbose=-1,
        )
        # Use a validation split for early stopping to prevent overfitting
        val_split = int(len(X_train) * 0.85)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[],  # LightGBM handles early stopping internally
        )
        lgb_ret_pred = lgb_model.predict(X_test)
        lgb_pred = close_test * (1 + lgb_ret_pred)
        lgb_metrics = compute_metrics(y_test_price, lgb_pred)
        lgb_extra = compute_return_metrics(y_test_ret, lgb_ret_pred)
        print(f"  RMSE={lgb_metrics['rmse']:.2f}  MAE={lgb_metrics['mae']:.2f}  R²={lgb_metrics['r2']:.4f}")

        models_list.append({"name": "LightGBM", "key": "lgbm", **lgb_metrics})
        extra_metrics["lgbm"] = lgb_extra
        chart_preds["LightGBM"] = lgb_pred
        trained_models["lgbm"] = lgb_model
    else:
        print("\n[7] Skipping LightGBM (not installed)")

    # ── Comparison tables ────────────────────────────────────────────────────
    print("\nPRICE metrics (Rs):")
    print("┌──────────────────────┬──────────┬──────────┬────────┐")
    print("│ Model                │   RMSE   │   MAE    │   R²   │")
    print("├──────────────────────┼──────────┼──────────┼────────┤")
    for m in models_list:
        print(f"│ {m['name']:<20} │ {m['rmse']:8.2f} │ {m['mae']:8.2f} │ {m['r2']:6.4f} │")
    print("└──────────────────────┴──────────┴──────────┴────────┘")

    print("\nRETURN metrics (the honest ones):")
    for key, label in [("naive", "Naive"), ("lr", "LR"), ("rf", "RF"),
                        ("xgb", "XGB"), ("lgbm", "LGBM")]:
        if key in extra_metrics:
            ex = extra_metrics[key]
            if ex.get("dir_acc") is not None:
                print(f"  {label:6s}: R²_return={ex['r2_return']:.4f}  direction accuracy={ex['dir_acc']*100:.1f}%")
            else:
                print(f"  {label:6s}: R²_return={ex['r2_return']:.4f}")

    # ── Feature importances (Random Forest) ──────────────────────────────────
    feature_importances = {
        feature: round(float(importance), 6)
        for feature, importance in zip(FEATURES, rf_model.feature_importances_)
    }
    print("\n  Random Forest feature importances:")
    for feat, imp in sorted(feature_importances.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.4f}")

    # ── XGBoost feature importances (if available) ───────────────────────────
    xgb_feature_importances = {}
    if _HAS_XGB and "xgb" in trained_models:
        xgb_feature_importances = {
            feature: round(float(importance), 6)
            for feature, importance in zip(FEATURES, xgb_model.feature_importances_)
        }
        print("\n  XGBoost feature importances:")
        for feat, imp in sorted(xgb_feature_importances.items(), key=lambda x: -x[1]):
            print(f"    {feat}: {imp:.4f}")

    # ── Save models ──────────────────────────────────────────────────────────
    print("\n[8] Saving models ...")
    joblib.dump(lr_model, MODEL_LR_PATH)
    print(f"  Saved LR   -> {MODEL_LR_PATH}")
    joblib.dump(rf_model, MODEL_RF_PATH)
    print(f"  Saved RF   -> {MODEL_RF_PATH}")

    if _HAS_XGB and "xgb" in trained_models:
        joblib.dump(xgb_model, MODEL_XGB_PATH)
        print(f"  Saved XGB  -> {MODEL_XGB_PATH}")

    if _HAS_LGB and "lgbm" in trained_models:
        joblib.dump(lgb_model, MODEL_LGB_PATH)
        print(f"  Saved LGBM -> {MODEL_LGB_PATH}")

    # ── Save metrics.json ────────────────────────────────────────────────────
    print("\n[9] Saving metrics.json ...")
    metrics_data = {
        "trained_on": "RELIANCE.NS",
        "train_date": str(date.today()),
        "features": FEATURES,
        "target": "next_day_return",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "models": models_list,
        "extra_metrics": extra_metrics,
        "feature_importances": feature_importances,
        "xgb_feature_importances": xgb_feature_importances,
        "best_params": best_params_all,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Saved metrics -> {METRICS_PATH}")

    print("\n[10] Saving comparison chart ...")
    save_comparison_chart(test_df, y_test_price, chart_preds)

    print("\n✅ Training complete.")
