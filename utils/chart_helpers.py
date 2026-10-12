"""
utils/chart_helpers.py
----------------------
Builds Plotly charts used on the Dashboard page.

We put chart code here (instead of in the page file) because:
  - It keeps pages/1_Dashboard.py short and easy to read
  - If we ever want to reuse a chart on another page, we just import it

Two functions live here:

  build_chart(history_data, symbol, chart_type="line")
    Main price chart. Two chart types:
      "line"   (default) - actual close price as a blue line
      "candle"           - candlestick (open/high/low/close)
    Both types also draw: Predicted close, SMA20, SMA50.

  build_indicator_chart(history_data, symbol)
    Three stacked subplots (shared x-axis):
      (a) Close price + Bollinger Bands
      (b) RSI with 30/70 reference lines
      (c) MACD line, signal line, histogram bars
    Expects the keys added in Phase 1: rsi, macd, macd_signal,
    macd_hist, bb_upper, bb_lower.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# ---------------------------------------------------------------------------
# Indicator chart  (RSI, MACD, Bollinger Bands)  — 3 stacked subplots
# ---------------------------------------------------------------------------

def build_indicator_chart(history_data: list, symbol: str) -> go.Figure:
    """
    Builds a 3-panel technical indicator chart for one stock.

    Why a separate function?
      The indicator chart has a very different structure from the main price
      chart (subplots, different y-axes, bar traces for MACD histogram) so
      keeping it separate makes both functions easier to read.

    Panels:
      Row 1 — Close price + Bollinger Bands (upper/lower envelope)
      Row 2 — RSI with horizontal reference lines at 30 and 70
      Row 3 — MACD line, Signal line, and MACD histogram bars

    history_data : list of dicts from get_history_with_predictions().
                   Expects keys: actual, bb_upper, bb_lower, rsi,
                   macd, macd_signal, macd_hist, date.
    symbol       : stock ticker string, used in the chart title.
    """
    df = pd.DataFrame(history_data)

    # Create figure with 3 rows sharing the same x-axis
    # Row heights: price panel is tallest, RSI and MACD share the rest
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=[
            f"{symbol} — Price & Bollinger Bands",
            "RSI (14)",
            "MACD (12, 26, 9)",
        ],
    )

    # ---- Row 1: Close price + Bollinger Bands --------------------------------

    # Bollinger Band fill between upper and lower
    # We draw the upper band first with no fill, then the lower band with
    # fill="tonexty" which shades the area between them.
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["bb_upper"],
            name="BB Upper",
            line=dict(color="rgba(100, 181, 246, 0.6)", width=1),
            hovertemplate="BB Upper: ₹%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["bb_lower"],
            name="BB Lower",
            line=dict(color="rgba(100, 181, 246, 0.6)", width=1),
            fill="tonexty",
            fillcolor="rgba(100, 181, 246, 0.08)",  # very light blue shading
            hovertemplate="BB Lower: ₹%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Close price on top of the band fill
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["actual"],
            name="Close",
            line=dict(color="#3B82F6", width=2),
            hovertemplate="Close: ₹%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ---- Row 2: RSI ----------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["rsi"],
            name="RSI 14",
            line=dict(color="#F59E0B", width=1.5),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ),
        row=2, col=1,
    )
    # Overbought reference line at 70
    fig.add_hline(
        y=70, line_dash="dot", line_color="rgba(239,68,68,0.6)",
        annotation_text="70", annotation_position="right",
        row=2, col=1,
    )
    # Oversold reference line at 30
    fig.add_hline(
        y=30, line_dash="dot", line_color="rgba(34,197,94,0.6)",
        annotation_text="30", annotation_position="right",
        row=2, col=1,
    )

    # ---- Row 3: MACD ---------------------------------------------------------

    # Histogram bars: green when MACD_Hist > 0 (bullish), red otherwise
    bar_colors = [
        "#22C55E" if (v is not None and v >= 0) else "#EF4444"
        for v in df["macd_hist"]
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["macd_hist"],
            name="MACD Histogram",
            marker_color=bar_colors,
            opacity=0.6,
            hovertemplate="Hist: %{y:.3f}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["macd"],
            name="MACD",
            line=dict(color="#3B82F6", width=1.5),
            hovertemplate="MACD: %{y:.3f}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["macd_signal"],
            name="Signal",
            line=dict(color="#F59E0B", width=1.5),
            hovertemplate="Signal: %{y:.3f}<extra></extra>",
        ),
        row=3, col=1,
    )

    # ---- Layout / styling ----------------------------------------------------

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=60, b=80),
    )

    # y-axis labels
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI",       row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",      row=3, col=1)

    # x-axis: nicer ticks + skip weekend gaps on ALL three panels
    fig.update_xaxes(
        tickformat="%b %Y",
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    return fig