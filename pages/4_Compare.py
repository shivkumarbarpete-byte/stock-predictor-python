"""
pages/4_Compare.py
------------------
Compare 2-4 NSE stocks side-by-side over the last 120 trading days.

What this page does:
  1. User types 2-4 stock symbols separated by commas (each must end with .NS).
  2. We validate symbols, fetch price history for each, and skip any that fail
     (bad symbol, no internet data) with a visible warning — we never crash.
  3. We show:
       - A Plotly line chart of % change from the first day of the 120-day
         window for each valid symbol (so all lines start at 0%, making
         comparison fair regardless of absolute price level).
       - A summary table: start price, end price, % change, max drawdown,
         annualised volatility.
       - A CSV download of the summary table.

No model / prediction is used on this page — it is purely price comparison.

Why normalise to % change?
  A stock at Rs 100 that grows to Rs 120 (+20%) is directly comparable to a
  stock at Rs 3000 that grows to Rs 3150 (+5%) only when both are expressed
  as percentage change from the same starting point.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_service.stock_utils import fetch_stock_data


# --- Page config ---
st.set_page_config(
    page_title="Compare Stocks — Stock Predictor",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Compare Stocks")
st.caption(
    "Enter 2 to 4 NSE symbols separated by commas (e.g. RELIANCE.NS, TCS.NS, INFY.NS). "
    "Each symbol must end with **.NS**. We show % change over the last 120 trading days."
)

st.divider()


# ---------------------------------------------------------------------------
# PURE HELPER FUNCTIONS
# (defined at the top so they are easy to read and unit-test independently)
# ---------------------------------------------------------------------------

def parse_symbols(raw_input: str) -> tuple:
    """
    Splits the raw text input into individual symbols and validates them.

    Why separate this from the UI?
      Validation logic is easier to test when it is a plain function that
      takes a string and returns two lists, rather than interleaved with
      Streamlit widgets.

    Returns:
      valid   : list of clean, upper-cased symbols ending with .NS
      invalid : list of tokens that failed validation (for error messages)
    """
    # Split on comma, strip whitespace, uppercase each token
    tokens = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

    valid   = []
    invalid = []

    for token in tokens:
        if not token.endswith(".NS"):
            invalid.append(token)
        else:
            valid.append(token)

    return valid, invalid


def compute_pct_change_series(df: pd.DataFrame, days: int = 120) -> pd.Series:
    """
    Takes a DataFrame with Date and Close columns (oldest to newest) and returns
    the % change from the first available close price over the last N rows.

    Example: if Day 1 close = 100 and Day 10 close = 115, the series shows
    0.0 on Day 1 and 15.0 on Day 10.

    Why use the first row of the slice (not the earliest row in history)?
      We want all stocks to start at 0% on the same calendar date (the
      start of the 120-day window) so the chart lines begin at the same
      point and are directly comparable.
    """
    recent = df.tail(days).copy().reset_index(drop=True)
    first_close = recent["Close"].iloc[0]
    pct = (recent["Close"] / first_close - 1) * 100
    pct.index = pd.to_datetime(recent["Date"])
    return pct


def compute_max_drawdown(close_series: pd.Series) -> float:
    """
    Computes the maximum drawdown of a price series as a percentage.

    Max drawdown = the largest peak-to-trough decline over the period.
    Example: if the price peaks at 200 then falls to 150, the drawdown is
    (200 - 150) / 200 = 25%.

    A higher (more negative) value means a bigger loss from peak.
    Returned as a negative percentage (e.g. -18.5 means -18.5% drawdown).
    """
    # Running maximum up to each point in time (the 'peak so far')
    running_max = close_series.cummax()

    # Drawdown at each day: how far below the running peak is the price?
    drawdown = (close_series - running_max) / running_max * 100

    # Max drawdown = the worst (most negative) value
    return round(float(drawdown.min()), 2)


def compute_annualised_volatility(close_series: pd.Series) -> float:
    """
    Computes annualised historical volatility as a percentage.

    Method: standard deviation of daily log returns * sqrt(252).
    252 = approximate number of trading days in a year.

    Log returns are used instead of simple returns because they are
    time-additive and better behaved statistically.
    """
    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    vol = float(log_returns.std() * np.sqrt(252) * 100)
    return round(vol, 2)


def build_compare_chart(symbol_series: dict) -> go.Figure:
    """
    Builds a Plotly line chart showing % change for multiple stocks.

    symbol_series : dict mapping symbol string -> pd.Series of % change
                    (index = DatetimeIndex, values = float % change from day 1)

    Why not use px.line?
      We add one trace per stock manually so we can control hover text,
      colours, and keep the same style conventions as the rest of the project.
    """
    # A small palette so lines are easily distinguishable
    COLORS = ["#3B82F6", "#F59E0B", "#22C55E", "#EF4444"]

    fig = go.Figure()

    for i, (symbol, series) in enumerate(symbol_series.items()):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=series.index.strftime("%Y-%m-%d"),
            y=series.values,
            name=symbol,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"{symbol}<br>Date: %{{x}}<br>Change: %{{y:.2f}}%"
                "<extra></extra>"
            ),
        ))

    # Zero reference line so the "flat" baseline is obvious
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(0,0,0,0.25)",
        annotation_text="0%",
        annotation_position="left",
    )

    fig.update_layout(
        title=dict(
            text="Normalised % Change from Day 1 (last 120 trading days)",
            font=dict(size=16),
        ),
        xaxis_title="Date",
        yaxis_title="% Change from First Day",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=60, b=80),
    )

    # Skip weekend gaps, same as the other charts in the project
    fig.update_xaxes(
        tickformat="%b %Y",
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    return fig


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

# We store the last comparison results so re-running the page (e.g. switching
# to another tab and back) does not immediately re-download everything.
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None   # will hold a dict of results


# ---------------------------------------------------------------------------
# INPUT CONTROLS
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([5, 1])

with col_input:
    raw_input = st.text_input(
        label="NSE Symbols (comma separated)",
        value="RELIANCE.NS, TCS.NS, INFY.NS",
        placeholder="e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS, WIPRO.NS",
        help="Enter 2 to 4 symbols. Each must end with .NS (NSE India only).",
    )

with col_btn:
    st.write("")   # vertical spacer to align button with input
    compare_clicked = st.button(
        "📊 Compare",
        use_container_width=True,
        type="primary",
    )


# ---------------------------------------------------------------------------
# VALIDATION + FETCH
# ---------------------------------------------------------------------------

if compare_clicked:

    valid_symbols, invalid_symbols = parse_symbols(raw_input)

    # --- Warn about tokens that failed format validation ---
    for bad in invalid_symbols:
        st.warning(
            f"⚠️ **'{bad}'** was ignored — NSE symbols must end with **.NS** "
            f"(e.g. {bad}.NS)."
        )

    # --- Need at least 2 valid symbols to compare ---
    if len(valid_symbols) < 2:
        st.error(
            "❌ Please enter at least **2 valid .NS symbols** to compare. "
            "Separate them with commas, e.g. `RELIANCE.NS, TCS.NS`."
        )
        st.session_state.compare_results = None
        st.stop()

    # --- Cap at 4 symbols so the chart stays readable ---
    if len(valid_symbols) > 4:
        st.info(
            f"ℹ️ More than 4 symbols entered. Using the first 4: "
            f"{', '.join(valid_symbols[:4])}."
        )
        valid_symbols = valid_symbols[:4]

    # --- Deduplicate while preserving order ---
    seen   = set()
    deduped = []
    for s in valid_symbols:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    valid_symbols = deduped

    # --- Fetch price history for each valid symbol ---
    results        = {}   # symbol -> dict of computed stats
    failed_symbols = []   # symbols that raised an exception

    with st.spinner("Fetching price data from Yahoo Finance…"):
        for sym in valid_symbols:
            try:
                df = fetch_stock_data(sym, period="1y")

                # Use however many days are available (up to 120)
                n_days = min(120, len(df))

                if n_days < 5:
                    raise ValueError(
                        f"Only {n_days} trading days of data are available. "
                        "Not enough to compare."
                    )

                # Build a Close series with DatetimeIndex for calculations
                recent_df    = df.tail(n_days).reset_index(drop=True)
                close_series = pd.Series(
                    recent_df["Close"].values,
                    index=pd.to_datetime(recent_df["Date"]),
                )

                results[sym] = {
                    "close":        close_series,
                    "pct":          compute_pct_change_series(df, days=n_days),
                    "start_price":  round(float(close_series.iloc[0]),  2),
                    "end_price":    round(float(close_series.iloc[-1]), 2),
                    "pct_change":   round(
                        float((close_series.iloc[-1] / close_series.iloc[0] - 1) * 100),
                        2,
                    ),
                    "max_drawdown": compute_max_drawdown(close_series),
                    "ann_vol":      compute_annualised_volatility(close_series),
                }

            except Exception as exc:
                # Catch everything: bad symbol, network error, empty data
                failed_symbols.append(sym)
                st.warning(f"⚠️ Skipping **{sym}**: {exc}")

    # After skipping failures, check we still have 2+ stocks
    if len(results) < 2:
        st.error(
            "❌ Not enough valid stocks to compare after data fetch "
            "(need at least 2). Check your symbols and internet connection."
        )
        st.session_state.compare_results = None
        st.stop()

    # Persist in session_state so the display block re-renders without re-fetching
    st.session_state.compare_results = results


# ---------------------------------------------------------------------------
# DISPLAY RESULTS — only rendered when session_state has data
# ---------------------------------------------------------------------------

if st.session_state.compare_results is not None:

    results     = st.session_state.compare_results
    symbol_list = list(results.keys())

    # ---- Normalised % change chart -------------------------------------------

    pct_dict = {sym: r["pct"] for sym, r in results.items()}
    fig = build_compare_chart(pct_dict)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Summary statistics table --------------------------------------------

    st.subheader("Summary Statistics (last 120 trading days)")
    st.caption(
        "**Max Drawdown**: largest peak-to-trough price decline (more negative = worse). "
        "**Ann. Volatility**: std of daily log-returns × √252 "
        "(higher = more volatile / higher risk)."
    )

    table_rows = []
    for sym, r in results.items():
        table_rows.append({
            "Symbol":              sym,
            "Start Price (Rs)":    r["start_price"],
            "End Price (Rs)":      r["end_price"],
            "% Change":            r["pct_change"],
            "Max Drawdown (%)":    r["max_drawdown"],
            "Ann. Volatility (%)": r["ann_vol"],
        })

    table_df = pd.DataFrame(table_rows)

    # Colour the % Change column: green = positive, red = negative
    def _colour_pct(val):
        """Returns CSS colour string so positive/negative changes are obvious."""
        color = "#22C55E" if val >= 0 else "#EF4444"
        return f"color: {color}; font-weight: bold;"

    styled = (
        table_df.style
        .map(_colour_pct, subset=["% Change"])
        .format({
            "Start Price (Rs)":    "{:,.2f}",
            "End Price (Rs)":      "{:,.2f}",
            "% Change":            "{:+.2f}%",
            "Max Drawdown (%)":    "{:.2f}%",
            "Ann. Volatility (%)": "{:.2f}%",
        })
        .set_properties(**{"text-align": "right"})
        .set_table_styles([{
            "selector": "th",
            "props": [("text-align", "center"), ("font-weight", "bold")],
        }])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ---- CSV download --------------------------------------------------------

    csv_str      = table_df.to_csv(index=False)
    symbols_slug = "_vs_".join(s.replace(".NS", "") for s in symbol_list)

    st.download_button(
        label="⬇️ Download summary as CSV",
        data=csv_str,
        file_name=f"compare_{symbols_slug}.csv",
        mime="text/csv",
        help="Downloads the summary statistics table as a CSV file.",
    )

    st.divider()

    # ---- Disclaimer ----------------------------------------------------------
    st.markdown(
        "_Data sourced from Yahoo Finance via yfinance. "
        "Past price performance is not a guarantee of future returns. "
        "This tool is for educational and portfolio-project purposes only._"
    )
