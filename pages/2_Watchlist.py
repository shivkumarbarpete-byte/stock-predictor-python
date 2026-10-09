"""
pages/2_Watchlist.py
--------------------
Shows the user's saved watchlist of stock symbols.

For each stock the user sees:
  - The stock symbol name
  - A "📊 View Chart" button — loads that stock's chart inline on this page
  - A "🗑️ Remove" button — removes it from the watchlist

The watchlist is loaded from data/watchlist.json via utils/watchlist.py.
"""

import streamlit as st

from ml_service.stock_utils import get_history_with_predictions, predict_next_close
from utils.watchlist import load_watchlist, save_watchlist
from utils.chart_helpers import build_chart


# --- Page config ---
st.set_page_config(
    page_title="Watchlist — Stock Predictor",
    page_icon="⭐",
    layout="wide",
)

st.title("⭐ My Watchlist")
st.caption("Your saved stocks. Go to the Dashboard to add new ones.")

st.divider()


# ---------------------------------------------------------------------------
# SESSION STATE SETUP
# We use session_state to remember which stock's chart is currently expanded.
# ---------------------------------------------------------------------------

# viewing_symbol : the symbol whose chart is currently shown (None = none shown)
if "viewing_symbol" not in st.session_state:
    st.session_state.viewing_symbol = None


# ---------------------------------------------------------------------------
# LOAD THE WATCHLIST
# ---------------------------------------------------------------------------

watchlist = load_watchlist()

if len(watchlist) == 0:
    # No stocks saved yet — show a helpful message
    st.info(
        "📭 Your watchlist is empty.\n\n"
        "Go to the **📊 Dashboard** page, search for a stock, and click "
        "**⭐ Add to Watchlist**."
    )
else:
    st.subheader(f"You have {len(watchlist)} stock(s) saved:")
    st.write("")  # small vertical space

    # Loop over every saved symbol and show a row with buttons
    for symbol in watchlist:

        # Use columns to lay out: symbol name | View Chart button | Remove button
        col_name, col_view, col_remove = st.columns([4, 2, 2])

        with col_name:
            st.markdown(f"### 🏷️ `{symbol}`")

        with col_view:
            # Unique key per button (Streamlit requires unique keys for buttons in loops)
            view_key = f"view_{symbol}"
            if st.button("📊 View Chart", key=view_key, use_container_width=True):
                # If user clicks this stock's View button again, toggle it off
                if st.session_state.viewing_symbol == symbol:
                    st.session_state.viewing_symbol = None
                else:
                    st.session_state.viewing_symbol = symbol
                # st.rerun() refreshes the page so the chart appears immediately
                st.rerun()

        with col_remove:
            remove_key = f"remove_{symbol}"
            if st.button("🗑️ Remove", key=remove_key, use_container_width=True):
                # Remove this symbol from the list
                watchlist.remove(symbol)
                save_watchlist(watchlist)

                # If we were viewing this stock's chart, close it
                if st.session_state.viewing_symbol == symbol:
                    st.session_state.viewing_symbol = None

                st.success(f"✅ **{symbol}** removed from your watchlist.")
                st.rerun()

        # --- Inline chart (shown when user clicks "View Chart") ---
        # We only show the chart for the symbol stored in session_state.viewing_symbol
        if st.session_state.viewing_symbol == symbol:
            with st.spinner(f"Loading chart for **{symbol}**…"):
                try:
                    # Fetch history and prediction for this stock
                    chart_data = get_history_with_predictions(symbol, days=120)
                    pred       = predict_next_close(symbol)

                    # --- Metric cards for this stock ---
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("💰 Current Close", f"₹{pred['last_close']:,.2f}")
                    with m2:
                        direction_emoji = "🟢" if pred["direction"] == "UP" else "🔴"
                        st.metric(
                            f"🔮 Predicted Tomorrow {direction_emoji}",
                            f"₹{pred['predicted_close']:,.2f}",
                            delta=f"₹{pred['change']:+.2f} ({pred['change_percent']:+.2f}%)",
                        )
                    with m3:
                        st.metric("📐 R² Score", f"{pred['metrics']['r2']:.4f}")

                    # --- Plotly chart ---
                    fig = build_chart(chart_data, symbol)
                    st.plotly_chart(fig, use_container_width=True)

                except ValueError as e:
                    # Wrong symbol or no data — show a friendly error
                    st.error(
                        f"❌ Could not load data for **{symbol}**.\n\n"
                        f"**Details:** {e}"
                    )
                except Exception as e:
                    st.error(
                        f"⚠️ Unexpected error loading **{symbol}**: {e}\n\n"
                        "Check your internet connection and try again."
                    )

        st.divider()  # separator between stocks


# ---------------------------------------------------------------------------
# Footer hint
# ---------------------------------------------------------------------------
st.markdown(
    "_To add more stocks, go to **📊 Dashboard**, search for a symbol, "
    "and click **⭐ Add to Watchlist**._"
)
