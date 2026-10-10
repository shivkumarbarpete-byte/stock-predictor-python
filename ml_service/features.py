"""
ml_service/features.py
----------------------
Builds SCALE-FREE features. Used by BOTH train_model.py and stock_utils.py
so training and prediction always create features the exact same way.

Idea: SMA20 / Close tells "is the 20-day average above or below today's
price, and by how much (in %)". It is ~1.0 for ANY stock, whatever its price.
"""

import pandas as pd


def add_scale_free_features(df: pd.DataFrame) -> pd.DataFrame:
    """Needs columns: Close, SMA20, SMA50, EMA20. Returns a copy with 3 new ratio columns."""
    df = df.copy()
    df["SMA20_ratio"] = df["SMA20"] / df["Close"]
    df["SMA50_ratio"] = df["SMA50"] / df["Close"]
    df["EMA20_ratio"] = df["EMA20"] / df["Close"]
    return df