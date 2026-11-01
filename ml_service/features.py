"""
ml_service/features.py
----------------------
Builds SCALE-FREE features. Used by BOTH train_model.py and stock_utils.py
so training and prediction always create features the exact same way.

Idea: SMA20 / Close tells "is the 20-day average above or below today's
price, and by how much (in %)". It is ~1.0 for ANY stock, whatever its price.

Sentiment_Score: aggregated VADER polarity from recent financial news headlines.
  - Range: roughly -1.0 (very bearish) to +1.0 (very bullish), 0.0 = neutral.
  - During LIVE prediction: fetched via news_sentiment.analyze_news_sentiment().
  - During TRAINING on historical CSV: defaults to 0.0 (neutral) because we
    don't have historical headlines archived. This is a known limitation;
    the model learns the other features' signal and treats sentiment as
    an additive boost at inference time.
"""

import pandas as pd


def add_scale_free_features(df: pd.DataFrame) -> pd.DataFrame:
    """Needs columns: Close, SMA20, SMA50, EMA20. Returns a copy with 3 new ratio columns."""
    df = df.copy()
    df["SMA20_ratio"] = df["SMA20"] / df["Close"]
    df["SMA50_ratio"] = df["SMA50"] / df["Close"]
    df["EMA20_ratio"] = df["EMA20"] / df["Close"]
    return df


def add_sentiment_feature(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Adds Sentiment_Score column to the DataFrame.

    Args:
        df: DataFrame with OHLCV + indicator columns.
        sentiment_score: A single float representing the aggregated news
            polarity. During training on historical data this defaults to 0.0
            (neutral). During live prediction it should be the real-time score
            from news_sentiment.analyze_news_sentiment().

    Returns:
        DataFrame with a new 'Sentiment_Score' column.
    """
    df = df.copy()
    df["Sentiment_Score"] = sentiment_score
    return df


def compute_live_sentiment_score(symbol: str) -> float:
    """
    Fetches live news and returns the aggregated sentiment polarity score.

    Falls back to 0.0 (neutral) if anything goes wrong — this ensures the
    prediction pipeline never crashes due to a news API failure.
    """
    try:
        # Lazy import to avoid circular dependencies and keep training fast
        try:
            from ml_service.news_sentiment import analyze_news_sentiment
        except ImportError:
            from news_sentiment import analyze_news_sentiment

        result = analyze_news_sentiment(symbol)
        return float(result.get("overall_score", 0.0))
    except Exception:
        return 0.0
