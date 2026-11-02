"""
ml_service/fetch_data.py
------------------------
Utility CLI script for downloading and saving historical OHLCV data from Yahoo Finance.
Reuses the cached fetch_stock_data from stock_utils to eliminate duplicate fetching logic.

Run with: python ml_service/fetch_data.py
"""

from pathlib import Path
import pandas as pd

try:
    from ml_service.stock_utils import fetch_stock_data
except ImportError:
    from stock_utils import fetch_stock_data


def save_to_csv(df: pd.DataFrame, symbol: str) -> Path:
    """Saves the DataFrame to a CSV file in the ml_service folder."""
    output_path = Path(__file__).parent / f"{symbol.replace('.NS', '')}_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    SYMBOL = "RELIANCE.NS"
    print(f"Fetching data for {SYMBOL} ...")
    data = fetch_stock_data(SYMBOL, period="2y", interval="1d")

    print("\n--- First 5 rows ---")
    print(data.head())

    print("\n--- Last 5 rows ---")
    print(data.tail())

    print(f"\nTotal rows fetched: {len(data)}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

    save_to_csv(data, SYMBOL)
