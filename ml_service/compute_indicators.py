"""
ml_service/compute_indicators.py
---------------------------------
Reads raw OHLCV data and adds technical indicator columns.

Indicators added:
  SMA20        — Simple Moving Average, last 20 days
  SMA50        — Simple Moving Average, last 50 days
  EMA20        — Exponential Moving Average, last 20 days (recent days weighted more)
  Trend_Signal — 1 if SMA20 > SMA50 (bullish), 0 otherwise
  RSI          — Relative Strength Index, 14-day period (momentum indicator)

What is RSI?
  RSI measures whether a stock is "overbought" (RSI > 70) or "oversold" (RSI < 30).
  It is calculated from the average gains vs average losses over 14 days.
  Range: 0 to 100. Widely used in technical analysis.
"""

import pandas as pd
from pathlib import Path


def load_data(csv_path: str) -> pd.DataFrame:
    """Reads a CSV file with a Date column and returns a DataFrame."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes bad rows:
    - Volume == 0 means no trading happened (holiday/data error) — drop those rows.
    - Rows with any missing values are dropped.
    - Sorts oldest-to-newest (important for time-series work).
    """
    before = len(df)

    df = df[df["Volume"] > 0]
    df = df.dropna()
    df = df.sort_values("Date").reset_index(drop=True)

    after = len(df)
    print(f"Cleaning: removed {before - after} rows (holidays / missing data)")

    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds SMA20, SMA50, EMA20 and Trend_Signal columns.

    The first 49 rows will have NaN in SMA50 (not enough history yet).
    Those rows are dropped before model training via dropna().
    """
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Trend signal: 1 = short-term MA above long-term MA (bullish crossover)
    df["Trend_Signal"] = (df["SMA20"] > df["SMA50"]).astype(int)

    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds the RSI (Relative Strength Index) column to the DataFrame.

    How RSI is calculated (step by step):
      1. diff()              -> daily price change (today's close - yesterday's close)
      2. clip(lower=0)       -> keep only the positive changes (gains)
         clip(upper=0) * -1  -> keep only the negative changes turned positive (losses)
      3. rolling(14).mean()  -> average gain and average loss over 14 days
      4. RS = avg_gain / avg_loss
      5. RSI = 100 - (100 / (1 + RS))

    RSI > 70 -> possibly overbought (price ran up fast, may reverse down)
    RSI < 30 -> possibly oversold  (price fell fast, may bounce up)

    The first 14 rows will have NaN (not enough history). They are dropped later.
    """
    # Step 1: daily change in closing price
    delta = df["Close"].diff()

    # Step 2: separate gains (positive changes) and losses (negative changes)
    gain = delta.clip(lower=0)           # negative changes become 0
    loss = (-delta).clip(lower=0)        # positive changes become 0, negatives flipped

    # Step 3: rolling 14-day average of gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Step 4: Relative Strength
    # Add a tiny number to avg_loss to avoid dividing by zero on consecutive up-days
    rs = avg_gain / (avg_loss + 1e-10)

    # Step 5: RSI formula
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


if __name__ == "__main__":
    # Run this file directly to add ALL indicators to your raw CSV.
    # Command: python ml_service/compute_indicators.py
    ML_DIR = Path(__file__).parent

    INPUT_CSV  = ML_DIR / "RELIANCE_data.csv"
    OUTPUT_CSV = ML_DIR / "RELIANCE_data_with_indicators.csv"

    data = load_data(INPUT_CSV)
    print(f"Loaded {len(data)} rows from {INPUT_CSV}")

    data = clean_data(data)
    data = add_moving_averages(data)
    data = add_rsi(data)

    print("\n--- Last 10 rows (with indicators) ---")
    print(data[["Date", "Close", "SMA20", "SMA50", "EMA20", "RSI", "Trend_Signal"]].tail(10))

    data.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(data)} rows with indicators to {OUTPUT_CSV}")
