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


from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_chart(
    history_data: list,
    symbol: str,
    chart_type: str = "line",
    rmse: Optional[float] = None,
    show_rangeslider: bool = False,
) -> go.Figure:
    """
    Builds and returns an interactive Plotly figure for one stock.

    Features:
      - Line or Candlestick view with color-coded bull/bear bodies
      - Actual close price, model predicted price, SMA20, SMA50
      - Optional prediction confidence ribbon (+/- RMSE)
      - Interactive range selector buttons (1M, 3M, 6M, YTD, ALL)
      - Crosshair spikes and unified hover card
    """
    df = pd.DataFrame(history_data)
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # --- Confidence Ribbon around Prediction (+/- RMSE) if provided ---
    if rmse is not None and rmse > 0 and "predicted" in df.columns:
        valid_preds = df["predicted"].dropna()
        if not valid_preds.empty:
            upper_bound = df["predicted"] + rmse
            lower_bound = df["predicted"] - rmse

            fig.add_trace(go.Scatter(
                x=df["date"],
                y=upper_bound,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=lower_bound,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(245, 158, 11, 0.12)",  # soft amber glow
                name=f"Prediction Uncertainty (± ₹{rmse:.2f})",
                hoverinfo="skip",
            ))

    # --- Main price series: line OR candlestick ---
    if chart_type == "candle":
        fig.add_trace(go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["actual"],
            name="OHLC Price",
            increasing_line_color="#16A34A",   # emerald green
            decreasing_line_color="#DC2626",   # red
        ))
        title_text = f"<b>{symbol}</b> — Candlestick & Predictions (Last {len(df)} Sessions)"
    else:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["actual"],
            name="Actual Close",
            line=dict(color="#2563EB", width=2.2),   # Tailwind blue-600
            hovertemplate="Actual Close: <b>₹%{y:,.2f}</b><extra></extra>",
        ))
        title_text = f"<b>{symbol}</b> — Actual vs Model Predicted Price (Last {len(df)} Sessions)"

    # --- Predicted Close Price ---
    pred_color = "#D97706" if chart_type == "candle" else "#EA580C"
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["predicted"],
        name="Predicted Close",
        line=dict(color=pred_color, width=2.0, dash="solid"),
        hovertemplate="Predicted: <b>₹%{y:,.2f}</b><extra></extra>",
    ))

    # --- SMA20 (cyan-blue, dashed) ---
    if "sma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["sma20"],
            name="SMA 20 (Short Trend)",
            line=dict(color="#0284C7", width=1.5, dash="dash"),
            hovertemplate="SMA20: ₹%{y:,.2f}<extra></extra>",
        ))

    # --- SMA50 (purple, dashed) ---
    if "sma50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["sma50"],
            name="SMA 50 (Medium Trend)",
            line=dict(color="#9333EA", width=1.5, dash="dash"),
            hovertemplate="SMA50: ₹%{y:,.2f}<extra></extra>",
        ))

    # --- Interactive Layout & Range Selector Buttons ---
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color="#0F172A")),
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        template="plotly_white",
        height=480,
        margin=dict(l=45, r=40, t=60, b=80),
    )

    # Crosshair and Range Selector buttons
    fig.update_xaxes(
        tickformat="%b %Y",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        showgrid=True,
        gridcolor="#F1F5F9",
        rangebreaks=[dict(bounds=["sat", "mon"])],
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            bgcolor="#F8FAFC",
            activecolor="#DBEAFE",
            font=dict(size=11, color="#1E293B"),
        ),
        rangeslider_visible=show_rangeslider,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        showspikes=True,
        spikemode="across",
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
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=45, r=40, t=60, b=80),
    )

    # y-axis labels
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1, showspikes=True, gridcolor="#F1F5F9")
    fig.update_yaxes(title_text="RSI",       row=2, col=1, range=[0, 100], showspikes=True, gridcolor="#F1F5F9")
    fig.update_yaxes(title_text="MACD",      row=3, col=1, showspikes=True, gridcolor="#F1F5F9")

    # x-axis: nicer ticks, crosshair spikes, and skip weekend gaps on ALL three panels
    fig.update_xaxes(
        tickformat="%b %Y",
        rangebreaks=[dict(bounds=["sat", "mon"])],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        gridcolor="#F1F5F9",
    )

    return fig