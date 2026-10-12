"""
pages/1_Dashboard.py
--------------------
Main analysis page. The user types a stock symbol, picks a model, clicks Fetch.

New in Steps 8b / 9b:
  - Model selectbox: choose Linear Regression or Random Forest
  - Prediction range: shows ₹X ± ₹RMSE so users understand uncertainty
  - RSI added to the indicators row
  - If model choice changes, we clear stale results and ask user to re-fetch

How session_state works in Streamlit:
  Streamlit re-runs this entire file on every click or keystroke.
  st.session_state is a dictionary that PERSISTS between those re-runs,
  so we use it to remember fetched data without calling Yahoo Finance again.
"""

import streamlit as st

from ml_service.stock_utils import get_history_with_predictions, predict_next_close
from utils.watchlist import load_watchlist, save_watchlist
from utils.chart_helpers import build_chart, build_indicator_chart


# --- Page config ---
st.set_page_config(
    page_title="Dashboard — Stock Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Stock Analysis Dashboard")
st.caption("Enter any NSE stock symbol (must end in .NS), pick a model, and click Fetch.")

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE — initialise all keys before use
# ──────────────────────────────────────────────────────────────────────────────

if "chart_data"      not in st.session_state: st.session_state.chart_data      = None
if "prediction"      not in st.session_state: st.session_state.prediction      = None
if "current_symbol"  not in st.session_state: st.session_state.current_symbol  = None
if "model_name_used" not in st.session_state: st.session_state.model_name_used = None


# ──────────────────────────────────────────────────────────────────────────────
# INPUT CONTROLS
# ──────────────────────────────────────────────────────────────────────────────

col_sym, col_model, col_btn = st.columns([3, 2, 1])

with col_sym:
    symbol_input = st.text_input(
        label="Stock Symbol",
        value="RELIANCE.NS",
        placeholder="e.g. TCS.NS, INFY.NS, HDFCBANK.NS",
        help="NSE stocks only. Must end with .NS",
    ).upper().strip()

with col_model:
    # Let the user pick which trained model to use for predictions
    model_choice = st.selectbox(
        label="Prediction Model",
        options=["Linear Regression", "Random Forest"],
        index=0,
        help=(
            "Linear Regression: simple, fast, easy to explain.\n"
            "Random Forest: 100 decision trees averaged — usually more accurate."
        ),
    )
    # Convert the display name to the short key used inside stock_utils.py
    model_name = "lr" if model_choice == "Linear Regression" else "rf"

with col_btn:
    st.write("")  # vertical space to align button with inputs
    fetch_clicked = st.button("🔍 Fetch & Predict", use_container_width=True, type="primary")


# ── If the user changes the model, clear old results so the chart matches ──
# We track which model was last used for the currently displayed results.
if (st.session_state.model_name_used is not None
        and model_name != st.session_state.model_name_used
        and st.session_state.chart_data is not None):
    st.session_state.chart_data     = None
    st.session_state.prediction     = None
    st.session_state.model_name_used = None
    st.info(f"Model changed to **{model_choice}**. Click **Fetch & Predict** to reload.")


# ──────────────────────────────────────────────────────────────────────────────
# FETCH DATA when button is clicked
# ──────────────────────────────────────────────────────────────────────────────

if fetch_clicked:

    if not symbol_input.endswith(".NS"):
        st.error(
            "❌ Symbol must end with **.NS** (e.g. RELIANCE.NS, TCS.NS). "
            "This tool supports NSE India stocks only."
        )
    else:
        with st.spinner(f"Fetching **{symbol_input}** using **{model_choice}**… (a few seconds)"):
            try:
                # get_history_with_predictions() -> list of dicts for the chart
                chart_data = get_history_with_predictions(
                    symbol_input, days=120, model_name=model_name
                )
                # predict_next_close() -> single prediction dict with metrics
                prediction = predict_next_close(symbol_input, model_name=model_name)

                # Store in session_state so results survive the next re-run
                st.session_state.chart_data      = chart_data
                st.session_state.prediction      = prediction
                st.session_state.current_symbol  = symbol_input
                st.session_state.model_name_used = model_name

            except FileNotFoundError as e:
                # Model .pkl file hasn't been created yet
                st.error(
                    f"❌ Model file not found.\n\n**Details:** {e}\n\n"
                    "Please run: `python ml_service/train_model.py` first."
                )
                st.session_state.chart_data = None
                st.session_state.prediction = None

            except ValueError as e:
                # Wrong symbol or no internet data
                st.error(
                    f"❌ Could not load data for **{symbol_input}**.\n\n"
                    f"**Details:** {e}\n\n"
                    "Check the symbol spelling and your internet connection."
                )
                st.session_state.chart_data = None
                st.session_state.prediction = None

            except Exception as e:
                # Catch-all for unexpected issues
                st.error(
                    f"⚠️ Something unexpected went wrong.\n\n"
                    f"**Details:** {e}\n\n"
                    "Check your internet connection and try again."
                )
                st.session_state.chart_data = None
                st.session_state.prediction = None


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS — only shown when we have data in session_state
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.chart_data is not None and st.session_state.prediction is not None:

    pred   = st.session_state.prediction      # shortcut variable
    symbol = st.session_state.current_symbol
    rmse   = pred["metrics"]["rmse"]           # used to compute the prediction range

    # Human-readable model name for display
    model_label = "Linear Regression" if pred["model_used"] == "lr" else "Random Forest"

    st.subheader(f"Results for {symbol}  ·  Model: {model_label}")

    # ── Row 1: core metrics ──────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("📅 Last Trading Date", pred["last_date"])

    with m2:
        st.metric("💰 Current Close Price", f"₹{pred['last_close']:,.2f}")

    with m3:
        direction_emoji = "🟢" if pred["direction"] == "UP" else "🔴"
        st.metric(
            label=f"🔮 Predicted Tomorrow {direction_emoji}",
            value=f"₹{pred['predicted_close']:,.2f}",
            delta=f"₹{pred['change']:+.2f}  ({pred['change_percent']:+.2f}%)",
        )

    with m4:
        st.metric(
            label="📐 R² on Price",
            value=f"{pred['metrics']['r2']:.4f}",
            help=(
                "R² measures how much of the variance in tomorrow's price the model explains. "
                "**This number is almost always very high (>0.95) for any stock model — "
                "even the naive 'tomorrow = today' baseline gets R² ≈ 0.99.** "
                "It is NOT a sign the model is useful for trading. "
                "See 'R² on Return' below for the honest metric."
            ),
        )

    # ── Prediction range ─────────────────────────────────────────────────────
    # RMSE = the typical prediction error on the test set.
    # Showing ± RMSE gives a realistic confidence band around the prediction.
    predicted_price = pred["predicted_close"]
    lower_bound     = round(predicted_price - rmse, 2)
    upper_bound     = round(predicted_price + rmse, 2)

    st.info(
        f"**Prediction range (\u00b1 RMSE):** \u20b9{lower_bound:,.2f}  \u2014  \u20b9{upper_bound:,.2f}\n\n"
        f"The model's typical error (RMSE) on this stock is \u20b9{rmse:,.2f}. "
        f"This range is a **rough indication based on past error only** \u2014 "
        f"the actual close tomorrow may fall anywhere, including well outside this range. "
        f"This is NOT a guaranteed trading signal \u2014 see **\U0001f916 Model Info** for the full disclaimer."
    )

    st.divider()

    # ── Plotly chart ─────────────────────────────────────────────────────────
    chart_style = st.radio(
        label="Chart style",
        options=["Line", "Candlestick"],
        index=0,
        horizontal=True,
        key="chart_type_choice",
        help="Line shows only the close price. Candlestick shows open, high, low and close for each day.",
    )
    chart_type = "candle" if chart_style == "Candlestick" else "line"

    fig = build_chart(st.session_state.chart_data, symbol, chart_type=chart_type)
    st.plotly_chart(fig, use_container_width=True)

    # ── Technical Indicators expander ────────────────────────────────────────
    # Renders only from session_state, so the expander opening/closing does
    # NOT trigger a re-fetch from Yahoo Finance.
    with st.expander("Technical indicators chart (RSI, MACD, Bollinger)"):
        ind_fig = build_indicator_chart(st.session_state.chart_data, symbol)
        st.plotly_chart(ind_fig, use_container_width=True)

        st.caption(
            "**Bollinger Bands** (top): price envelope based on 20-day SMA ± 2 standard deviations. "
            "When price touches the upper band it may be overbought; lower band may indicate oversold.\n\n"
            "**RSI** (middle): momentum indicator (0–100). "
            "Above 70 = possibly overbought. Below 30 = possibly oversold.\n\n"
            "**MACD** (bottom): difference between 12-day and 26-day EMA. "
            "Green bars = MACD above signal (bullish momentum). Red bars = below signal (bearish)."
        )

    # ── CSV download ─────────────────────────────────────────────────────────
    # Convert the chart data list-of-dicts to a DataFrame, then to CSV bytes.
    # We build the CSV here so it is always in sync with what is displayed.
    import pandas as pd  # already a dependency; import inside block to keep top clean

    csv_df  = pd.DataFrame(st.session_state.chart_data)
    csv_str = csv_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download data as CSV",
        data=csv_str,
        file_name=f"{symbol}_history.csv",
        mime="text/csv",
        help="Downloads the last 120 days of OHLCV data plus all computed indicators as a CSV file.",
    )

    st.divider()

    # ── Technical indicators row ─────────────────────────────────────────────
    st.subheader("Technical Indicators (Latest Day)")

    ind = pred["indicators"]
    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.metric("SMA 20", f"₹{ind['SMA20']:,.2f}",
                  help="Simple Moving Average of last 20 trading days.")
    with i2:
        st.metric("SMA 50", f"₹{ind['SMA50']:,.2f}",
                  help="Simple Moving Average of last 50 trading days.")
    with i3:
        st.metric("EMA 20", f"₹{ind['EMA20']:,.2f}",
                  help="Exponential Moving Average — recent days count more.")
    with i4:
        rsi_val = ind.get("RSI", 0)
        # Colour the RSI label based on overbought/oversold thresholds
        rsi_status = "🔴 Overbought" if rsi_val > 70 else ("🟢 Oversold" if rsi_val < 30 else "⚪ Neutral")
        st.metric(
            f"RSI 14  ({rsi_status})",
            f"{rsi_val:.1f}",
            help="RSI > 70 = possibly overbought. RSI < 30 = possibly oversold.",
        )

    st.divider()

    # ── Model quality metrics row ─────────────────────────────────────────────
    st.subheader("Model Quality on This Stock (live test)")
    st.caption(
        "Computed on the last 20% of fetched data for this specific stock. "
        "Each metric is shown alongside the **Naive baseline** (tomorrow = today) so you can "
        "see whether the model actually improves on the simplest possible prediction."
    )

    m = pred["metrics"]

    # Row A: MAE and RMSE — model vs naive side by side
    st.markdown("**Error in price (\u20b9) \u2014 model vs naive baseline**")
    qa1, qa2, qa3, qa4 = st.columns(4)

    with qa1:
        st.metric(
            "MAE — Model", f"\u20b9{m['mae']:,.2f}",
            help="Mean Absolute Error: average prediction error in rupees (model).",
        )
    with qa2:
        st.metric(
            "MAE — Naive", f"\u20b9{m['naive_mae']:,.2f}",
            delta=f"{m['mae'] - m['naive_mae']:+.2f} vs naive",
            delta_color="inverse",
            help="Naive baseline: predict tomorrow = today. Negative delta means the model is WORSE.",
        )
    with qa3:
        st.metric(
            "RMSE — Model", f"\u20b9{m['rmse']:,.2f}",
            help="Root Mean Squared Error (model). Penalises large errors more than MAE.",
        )
    with qa4:
        st.metric(
            "RMSE — Naive", f"\u20b9{m['naive_rmse']:,.2f}",
            delta=f"{m['rmse'] - m['naive_rmse']:+.2f} vs naive",
            delta_color="inverse",
            help="RMSE for the naive baseline. Negative delta means the model is WORSE.",
        )

    # Row B: R2 scores and directional accuracy
    st.markdown("**R\u00b2 and directional accuracy**")
    qb1, qb2, qb3 = st.columns(3)

    with qb1:
        st.metric(
            "R\u00b2 on Price",
            f"{m['r2']:.4f}",
            help=(
                "Almost always close to 1.0 for any stock model, including the naive baseline. "
                "High price R\u00b2 does NOT mean the model is good \u2014 it just means tomorrow's price "
                "is very close to today's price (high autocorrelation)."
            ),
        )
    with qb2:
        st.metric(
            "R\u00b2 on Return",
            f"{m['r2_return']:.4f}",
            help=(
                "R\u00b2 computed on the predicted vs actual next-day RETURN (not price). "
                "This is the honest metric. A value near 0 or negative means the model "
                "explains essentially none of the day-to-day return variation. "
                "This is typical and expected for short-term stock models."
            ),
        )
    with qb3:
        dir_acc = m.get("directional_accuracy", 0)
        st.metric(
            "Directional Accuracy",
            f"{dir_acc:.1f}%",
            delta=f"{dir_acc - 50:.1f}pp vs 50% coin flip",
            delta_color="normal",
            help=(
                "% of test days where the model correctly predicted Up vs Down. "
                "50% = random coin flip. A good model should consistently beat 50%. "
                "Being close to 50% (or below) means the model has no real directional edge."
            ),
        )

    st.divider()

    # ── Add to Watchlist ──────────────────────────────────────────────────────
    watchlist = load_watchlist()

    if symbol in watchlist:
        st.success(f"✅ **{symbol}** is already in your watchlist.")
    else:
        if st.button(f"⭐ Add {symbol} to Watchlist"):
            watchlist.append(symbol)
            save_watchlist(watchlist)
            st.success(f"✅ **{symbol}** added to your watchlist!")
            st.rerun()
