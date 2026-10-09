"""
ml_service/stock_utils.py
--------------------------
All reusable ML logic — Streamlit pages import from here.

Five things this file does:
  1. fetch_stock_data()             -> download price history from Yahoo Finance
  2. add_indicators()               -> compute SMA20, SMA50, EMA20, RSI
  3. load_model(model_name)         -> load a saved .pkl model from disk
  4. predict_next_close(symbol, model_name) -> full pipeline, returns a prediction dict
  5. get_history_with_predictions() -> actual + predicted prices for the chart

IMPORTANT — Feature list:
  FEATURES is imported from ml_service/config.py.
  train_model.py also imports it from the same place.
  This guarantees training and prediction always use the SAME features.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the single source of truth for features and paths.
# Try/except handles two cases:
#   "from ml_service.config" works when Streamlit runs from the project root.
#   "from config" works when you run this script directly (python ml_service/stock_utils.py).
try:
    from ml_service.config import FEATURES, MODEL_LR_PATH, MODEL_RF_PATH, METRICS_PATH
    from ml_service.compute_indicators import add_rsi
except ImportError:
    from config import FEATURES, MODEL_LR_PATH, MODEL_RF_PATH, METRICS_PATH
    from compute_indicators import add_rsi


# -----------------------------------------------------------------------------
# 1. DATA FETCHING  (with 10-minute cache)
# -----------------------------------------------------------------------------

# _cache stores {cache_key: (dataframe, timestamp)} so we don't re-download
_cache = {}
CACHE_TTL = 600  # seconds (10 minutes)


def fetch_stock_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Downloads historical OHLCV data from Yahoo Finance for one stock.
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume.

    Uses an in-memory cache — if you call this twice within 10 minutes for
    the same symbol, the second call returns the cached copy instantly.
    """
    cache_key = f"{symbol}_{period}_{interval}"
    now = time.time()

    # Return cached copy if it's less than 10 minutes old
    if cache_key in _cache:
        cached_df, timestamp = _cache[cache_key]
        if now - timestamp < CACHE_TTL:
            return cached_df.copy()

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
    except Exception:
        raise ValueError(f"Failed to fetch data for '{symbol}'. Check your internet connection.")

    if df.empty:
        raise ValueError(
            f"No data found for symbol '{symbol}'. "
            "Is the spelling correct? NSE stocks need .NS (e.g. TCS.NS)."
        )

    df = df.reset_index()

    # yfinance returns timezone-aware dates. We strip the timezone so the
    # dates are plain datetime objects — easier to display and compare.
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    final_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    _cache[cache_key] = (final_df.copy(), now)

    return final_df


# -----------------------------------------------------------------------------
# 2. INDICATORS
# -----------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds SMA20, SMA50, EMA20, and RSI columns to the DataFrame.
    Works on a COPY so the original is never modified.

    After this, the first ~50 rows will have NaN values (SMA50 needs 50 rows
    of history). Always call df.dropna(subset=FEATURES) afterwards.
    """
    df = df.copy()  # never modify the original — this is good Python practice

    # Simple Moving Averages
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # Exponential Moving Average — recent days count more
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI — momentum indicator. add_rsi() is imported from compute_indicators.py
    df = add_rsi(df, period=14)

    return df


# -----------------------------------------------------------------------------
# 3. MODEL LOADING  (loads once, then caches in memory)
# -----------------------------------------------------------------------------

# _models is a dictionary: {"lr": <model object>, "rf": <model object>}
# We load each model only the first time it is requested.
_models = {}


def load_model(model_name: str = "lr"):
    """
    Loads the trained model from disk. Returns the cached version on repeat calls.

    model_name : "lr" for Linear Regression, "rf" for Random Forest
    The model files live inside ml_service/ (defined in config.py).
    """
    if model_name not in _models:
        # Pick the right file path based on which model is requested
        if model_name == "lr":
            path = MODEL_LR_PATH
        elif model_name == "rf":
            path = MODEL_RF_PATH
        else:
            raise ValueError(f"Unknown model '{model_name}'. Choose 'lr' or 'rf'.")

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Please run: python ml_service/train_model.py"
            )

        _models[model_name] = joblib.load(path)

    return _models[model_name]


# -----------------------------------------------------------------------------
# 4. PREDICTION
# -----------------------------------------------------------------------------

def predict_next_close(symbol: str, model_name: str = "lr") -> dict:
    """
    Full pipeline for one prediction:
      fetch -> add indicators -> drop NaN rows -> load model -> predict.

    Returns a dictionary with prediction, model quality metrics, and indicators.
    The model_name parameter lets the caller choose "lr" or "rf".
    """
    df = fetch_stock_data(symbol, period="1y")   # 1 year gives enough rows for SMA50 + RSI
    df = add_indicators(df)
    df = df.dropna(subset=FEATURES)              # drop the first ~50 rows with NaN

    if df.empty:
        raise ValueError(f"Not enough data for '{symbol}' to compute all indicators.")

    latest = df.iloc[-1]   # the most recent trading day

    # Model expects shape (1, n_features) — reshape the latest row's values
    X_latest = latest[FEATURES].values.reshape(1, -1)

    model = load_model(model_name)
    predicted = float(model.predict(X_latest)[0])

    last_close = float(latest["Close"])
    change     = predicted - last_close
    change_pct = (change / last_close) * 100

    # --- Compute dynamic metrics on the last 20% of fetched data ---
    # This tells us how accurate this model is for THIS specific stock recently.
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    test_df["Target_Close"] = test_df["Close"].shift(-1)
    test_df = test_df.dropna(subset=["Target_Close"])

    if not test_df.empty:
        X_test  = test_df[FEATURES].values
        y_test  = test_df["Target_Close"].values
        preds   = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}
    else:
        metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    return {
        "symbol":          symbol.upper(),
        "last_date":       latest["Date"].strftime("%Y-%m-%d"),
        "last_close":      round(last_close, 2),
        "predicted_close": round(predicted, 2),
        "change":          round(change, 2),
        "change_percent":  round(change_pct, 2),
        "direction":       "UP" if change > 0 else "DOWN",
        "model_used":      model_name,
        "indicators": {
            "SMA20": round(float(latest["SMA20"]), 2),
            "SMA50": round(float(latest["SMA50"]), 2),
            "EMA20": round(float(latest["EMA20"]), 2),
            "RSI":   round(float(latest["RSI"]),   2),
        },
        "metrics": metrics,
    }


# -----------------------------------------------------------------------------
# 5. HISTORY + PREDICTIONS  (for the Plotly chart)
# -----------------------------------------------------------------------------

def get_history_with_predictions(
    symbol: str,
    days: int = 120,
    model_name: str = "lr"
) -> list:
    """
    Returns the last N trading days of actual prices AND model predictions.
    Streamlit passes this list to build_chart() to draw the Plotly chart.

    Each item in the returned list looks like:
      { "date": "2024-01-15", "actual": 2450.0, "predicted": 2455.0,
        "sma20": 2440.0, "sma50": 2420.0 }

    Why shift(1)?
      Row i's features predict row i+1's price. shift(1) moves each
      prediction to align it visually with the day it is predicting.
    """
    df = fetch_stock_data(symbol, period="2y")
    df = add_indicators(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Not enough data for '{symbol}'.")

    model = load_model(model_name)
    df["Predicted"] = model.predict(df[FEATURES].values)

    # Shift predictions forward by one day so they align with the predicted date
    df["PredictedForToday"] = df["Predicted"].shift(1)

    recent = df.tail(days)

    result = []
    for _, row in recent.iterrows():
        pred = row["PredictedForToday"]
        result.append({
            "date":      row["Date"].strftime("%Y-%m-%d"),
            "actual":    round(float(row["Close"]), 2),
            "predicted": None if pd.isna(pred) else round(float(pred), 2),
            "sma20":     round(float(row["SMA20"]), 2),
            "sma50":     round(float(row["SMA50"]), 2),
        })
    return result


# -----------------------------------------------------------------------------
# Quick test: python ml_service/stock_utils.py
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing with RELIANCE.NS ...\n")
    result = predict_next_close("RELIANCE.NS", model_name="lr")
    for key, value in result.items():
        print(f"  {key}: {value}")
