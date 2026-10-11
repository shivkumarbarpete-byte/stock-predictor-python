"""
utils/chart_helpers.py
----------------------
Builds the Plotly chart used on the Dashboard page.

We put chart code here (instead of in the page file) because:
  - It keeps pages/1_Dashboard.py short and easy to read
  - If we ever want to reuse the chart on another page, we just import it

build_chart() takes the list returned by get_history_with_predictions()
and returns a Plotly figure object.

Two chart types (chart_type argument):
  "line"   (default) - actual close price as a blue line
  "candle"           - candlestick (open/high/low/close); green candle = close
                       above open, red candle = close below open

Both types also draw:
  - Predicted close price   - red,    solid
  - SMA20 (20-day average)  - green,  dashed
  - SMA50 (50-day average)  - purple, dashed
"""

import pandas as pd
import plotly.graph_objects as go


def build_chart(history_data: list, symbol: str, chart_type: str = "line") -> go.Figure:
    """
    Builds and returns a Plotly figure for one stock.

    history_data : list of dicts from get_history_with_predictions()
                   Each dict has keys: date, open, high, low, actual,
                   predicted, sma20, sma50
    symbol       : stock ticker (e.g. "RELIANCE.NS") - used in the chart title
    chart_type   : "line" (default) or "candle"
    """
    df = pd.DataFrame(history_data)

    fig = go.Figure()

    # --- Main price series: line OR candlestick ---
    if chart_type == "candle":
        fig.add_trace(go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["actual"],
            name="Price",
            increasing_line_color="#22C55E",   # green candle: close > open
            decreasing_line_color="#EF4444",   # red candle:   close < open
        ))
        title_text = f"{symbol} - Candlestick with Predicted Close (Last 120 days)"
    else:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["actual"],
            name="Actual Close",
            line=dict(color="#3B82F6", width=2),   # blue
            hovertemplate="Date: %{x}<br>Actual: ₹%{y:.2f}<extra></extra>",
        ))
        title_text = f"{symbol} - Actual vs Predicted Close Price (Last 120 days)"

    # --- Predicted Close Price (orange so it stands out on candles) ---
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["predicted"],
        name="Predicted Close",
        line=dict(color="#F59E0B" if chart_type == "candle" else "#EF4444", width=2),
        hovertemplate="Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>",
    ))

    # --- SMA20 (green, dashed) ---
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["sma20"],
        name="SMA 20",
        line=dict(color="#0EA5E9", width=1.5, dash="dash"),
        hovertemplate="Date: %{x}<br>SMA20: ₹%{y:.2f}<extra></extra>",
    ))

    # --- SMA50 (purple, dashed) ---
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["sma50"],
        name="SMA 50",
        line=dict(color="#A855F7", width=1.5, dash="dash"),
        hovertemplate="Date: %{x}<br>SMA50: ₹%{y:.2f}<extra></extra>",
    ))

    # --- Layout / styling ---
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=60, b=80),
        xaxis_rangeslider_visible=False,   # hide the slider Plotly adds under candlesticks
    )

    # Nicer date ticks + skip Saturday/Sunday gaps (market is closed)
    fig.update_xaxes(
        tickformat="%b %Y",
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    return fig