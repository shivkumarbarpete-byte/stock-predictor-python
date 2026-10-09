"""
utils/chart_helpers.py
----------------------
Builds the Plotly line chart used on the Dashboard page.

We put chart code here (instead of in the page file) because:
  - It keeps pages/1_Dashboard.py short and easy to read
  - If we ever want to reuse the chart on another page, we just import it

The function build_chart() takes the list returned by
get_history_with_predictions() and returns a Plotly figure object.

Chart has 4 lines:
  1. Actual close price      — blue,   solid
  2. Predicted close price   — red,    solid
  3. SMA20 (20-day average)  — green,  dashed
  4. SMA50 (50-day average)  — purple, dashed
"""

import pandas as pd
import plotly.graph_objects as go


def build_chart(history_data: list, symbol: str) -> go.Figure:
    """
    Builds and returns a Plotly figure with 4 lines for one stock.

    history_data : list of dicts from get_history_with_predictions()
                   Each dict has keys: date, actual, predicted, sma20, sma50
    symbol       : stock ticker (e.g. "RELIANCE.NS") — used in the chart title
    """
    # Convert the list of dicts to a DataFrame — easier to work with for charts
    df = pd.DataFrame(history_data)

    # Create an empty Plotly figure
    fig = go.Figure()

    # --- Line 1: Actual Close Price (blue, solid) ---
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["actual"],
        name="Actual Close",
        line=dict(color="#3B82F6", width=2),   # blue
        hovertemplate="Date: %{x}<br>Actual: ₹%{y:.2f}<extra></extra>",
    ))

    # --- Line 2: Predicted Close Price (red, solid) ---
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["predicted"],
        name="Predicted Close",
        line=dict(color="#EF4444", width=2),   # red
        hovertemplate="Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>",
    ))

    # --- Line 3: SMA20 — 20-day Simple Moving Average (green, dashed) ---
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["sma20"],
        name="SMA 20",
        line=dict(color="#22C55E", width=1.5, dash="dash"),   # green dashed
        hovertemplate="Date: %{x}<br>SMA20: ₹%{y:.2f}<extra></extra>",
    ))

    # --- Line 4: SMA50 — 50-day Simple Moving Average (purple, dashed) ---
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["sma50"],
        name="SMA 50",
        line=dict(color="#A855F7", width=1.5, dash="dash"),   # purple dashed
        hovertemplate="Date: %{x}<br>SMA50: ₹%{y:.2f}<extra></extra>",
    ))

    # --- Chart layout / styling ---
    fig.update_layout(
        title=dict(
            text=f"{symbol} — Actual vs Predicted Close Price (Last 120 days)",
            font=dict(size=16),
        ),
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",       # shows all 4 values when you hover over a date
        legend=dict(
            orientation="h",         # horizontal legend below the chart
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",     # clean white background
        height=450,
        margin=dict(l=40, r=40, t=60, b=80),
    )

    # Nicer date tick format on the x-axis
    fig.update_xaxes(tickformat="%b %Y")

    return fig
