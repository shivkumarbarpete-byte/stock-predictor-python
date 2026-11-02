"""
pages/2_Watchlist.py
--------------------
Interactive Watchlist & Portfolio Tracker.

Features:
  - Add and remove custom stock tickers with automatic format validation.
  - Track live current close, model predicted return, direction, RSI, and MA trend.
  - Optional portfolio tracking: enter shares owned and buy price to view total portfolio
    valuation and unrealized P&L in real time.
  - Interactive Plotly chart view with Candlestick/Line modes and range selector buttons.
  - Professional PDF and CSV export for portfolio records.
"""

import pandas as pd
import streamlit as st

from ml_service.stock_utils import get_history_with_predictions, predict_next_close
from ml_service.validation import validate_ticker_symbol, safe_fetch_stock_summary
from ml_service.pdf_report import generate_watchlist_pdf
from utils.watchlist import (
    load_watchlist_records, save_watchlist_records,
    add_to_watchlist, remove_from_watchlist,
)
from utils.chart_helpers import build_chart

# --- Page config ---
st.set_page_config(
    page_title="Watchlist & Portfolio — Stock Predictor",
    page_icon="⭐",
    layout="wide",
)

st.title("⭐ Watchlist & Portfolio Tracker")
st.caption(
    "Monitor your favorite NSE stocks, track model-forecasted price movements, "
    "and calculate portfolio returns in real time."
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# 1. ADD NEW STOCK TO WATCHLIST WIDGET
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("➕ Add Stock to Watchlist / Track Portfolio Holding", expanded=False):
    col_add_sym, col_add_shares, col_add_price, col_add_btn = st.columns([3, 2, 2, 2])

    with col_add_sym:
        new_symbol_raw = st.text_input(
            "Stock Symbol",
            placeholder="e.g. INFY or INFY.NS",
            help="NSE stock symbol (will auto-append .NS if omitted)",
            key="add_sym_input",
        )

    with col_add_shares:
        shares_input = st.number_input(
            "Shares Owned (Optional)",
            min_value=0,
            value=0,
            step=1,
            help="Enter number of shares to track investment capital and P&L.",
            key="add_shares_input",
        )

    with col_add_price:
        buy_price_input = st.number_input(
            "Avg Buy Price (₹) (Optional)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            format="%.2f",
            help="Average purchase price per share.",
            key="add_price_input",
        )

    with col_add_btn:
        st.write("")  # align vertically
        st.write("")
        add_clicked = st.button("⭐ Add / Update", use_container_width=True, type="primary")

    if add_clicked:
        is_valid, clean_sym, msg = validate_ticker_symbol(new_symbol_raw)
        if not is_valid:
            st.error(f"❌ {msg}")
        else:
            if msg:
                st.info(f"💡 {msg}")
            added = add_to_watchlist(clean_sym, shares=shares_input, buy_price=buy_price_input)
            if added:
                st.success(f"✅ Added **{clean_sym}** to your watchlist!")
            else:
                st.success(f"✅ Updated portfolio details for **{clean_sym}**!")
            st.rerun()

    # Quick NIFTY 50 suggestions pills
    st.caption("Quick Add Popular NIFTY 50 Stocks:")
    popular_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS"]
    p_cols = st.columns(len(popular_tickers))
    for idx, pop_sym in enumerate(popular_tickers):
        with p_cols[idx]:
            if st.button(f"+ {pop_sym.replace('.NS', '')}", key=f"quick_add_{pop_sym}", use_container_width=True):
                add_to_watchlist(pop_sym)
                st.success(f"Added **{pop_sym}** to watchlist!")
                st.rerun()

st.write("")

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD RECORDS & LIVE DATA SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
records = load_watchlist_records()

if not records:
    st.info(
        "📭 **Your watchlist is currently empty.**\n\n"
        "Click **➕ Add Stock to Watchlist** above or go to the **📊 Dashboard** to add stocks."
    )
    st.stop()

# State for expanded chart viewing
if "watchlist_view_sym" not in st.session_state:
    st.session_state.watchlist_view_sym = None

# Fetch live stats for each stock
tracker_data = []
total_invested = 0.0
total_current_val = 0.0
has_holdings = False

with st.spinner("Updating live quotes & predictions for watchlist stocks…"):
    for item in records:
        sym = item["symbol"]
        shares = item.get("shares", 0)
        buy_p = item.get("buy_price", 0.0)

        summary = safe_fetch_stock_summary(sym)
        if summary:
            curr_close = summary["current_close"]
            pred_close = summary["predicted_close"]
            chg_pct = summary["predicted_change_pct"]
            direction = summary["direction"]

            # Holding calculations
            if shares > 0:
                has_holdings = True
                cost_basis = shares * (buy_p if buy_p > 0 else curr_close)
                current_val = shares * curr_close
                pnl = current_val - cost_basis
                pnl_pct = (pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0

                total_invested += cost_basis
                total_current_val += current_val
            else:
                cost_basis = 0.0
                current_val = 0.0
                pnl = 0.0
                pnl_pct = 0.0

            tracker_data.append({
                "symbol": sym,
                "current_close": curr_close,
                "predicted_close": pred_close,
                "predicted_change_pct": chg_pct,
                "direction": direction,
                "rsi": summary["rsi"],
                "rsi_status": summary["rsi_status"],
                "trend": summary["trend"],
                "shares": shares,
                "buy_price": buy_p,
                "cost_basis": cost_basis,
                "current_val": current_val,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })
        else:
            # Fallback for failing ticker
            tracker_data.append({
                "symbol": sym,
                "current_close": 0.0,
                "predicted_close": 0.0,
                "predicted_change_pct": 0.0,
                "direction": "N/A",
                "rsi": 50.0,
                "rsi_status": "Unavailable",
                "trend": "N/A",
                "shares": shares,
                "buy_price": buy_p,
                "cost_basis": 0.0,
                "current_val": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
            })

# ──────────────────────────────────────────────────────────────────────────────
# 3. PORTFOLIO SUMMARY CARDS (If user configured share quantities)
# ──────────────────────────────────────────────────────────────────────────────
if has_holdings and total_invested > 0:
    st.subheader("💼 Active Portfolio Summary")
    unrealized_pnl = total_current_val - total_invested
    unrealized_pct = (unrealized_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.metric("Total Invested", f"₹{total_invested:,.2f}")
    with pc2:
        st.metric("Current Valuation", f"₹{total_current_val:,.2f}")
    with pc3:
        pnl_delta = f"{unrealized_pct:+.2f}%"
        st.metric(
            "Unrealized P&L",
            f"₹{unrealized_pnl:+,.2f}",
            delta=pnl_delta,
            delta_color="normal" if unrealized_pnl >= 0 else "inverse",
        )
    with pc4:
        st.metric("Tracked Holdings", f"{len([t for t in tracker_data if t['shares'] > 0])} stocks")

    st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# 4. WATCHLIST TRACKER TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"📋 Monitored Stocks ({len(tracker_data)})")

# Build formatted dataframe for summary view
table_rows = []
for t in tracker_data:
    row = {
        "Symbol": t["symbol"],
        "Current Price": f"₹{t['current_close']:,.2f}" if t['current_close'] > 0 else "—",
        "Predicted Close": f"₹{t['predicted_close']:,.2f}" if t['predicted_close'] > 0 else "—",
        "Exp. Return": f"{t['predicted_change_pct']:+.2f}%",
        "Signal": "🟢 UP" if t['direction'] == "UP" else ("🔴 DOWN" if t['direction'] == "DOWN" else "—"),
        "RSI (14)": f"{t['rsi']:.1f} ({t['rsi_status']})",
        "Trend Signal": t["trend"],
    }
    if has_holdings:
        row["Shares"] = t["shares"] if t["shares"] > 0 else "—"
        row["P&L (₹)"] = f"₹{t['pnl']:+,.2f} ({t['pnl_pct']:+.1f}%)" if t["shares"] > 0 else "—"

    table_rows.append(row)

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.markdown("")

# ──────────────────────────────────────────────────────────────────────────────
# 5. INDIVIDUAL STOCK ACTIONS & INTERACTIVE CHARTS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("#### 🔍 Stock Actions & Detailed Visualizations")

for item in tracker_data:
    sym = item["symbol"]
    col_sym_info, col_btn_chart, col_btn_del = st.columns([4, 2, 1.5])

    with col_sym_info:
        st.markdown(
            f"**`{sym}`** · Last: **₹{item['current_close']:,.2f}** · "
            f"Predicted: **₹{item['predicted_close']:,.2f}** ({item['predicted_change_pct']:+.2f}%)"
        )

    with col_btn_chart:
        chart_btn_label = "Hide Chart" if st.session_state.watchlist_view_sym == sym else "📊 View Chart"
        if st.button(chart_btn_label, key=f"view_btn_{sym}", use_container_width=True):
            if st.session_state.watchlist_view_sym == sym:
                st.session_state.watchlist_view_sym = None
            else:
                st.session_state.watchlist_view_sym = sym
            st.rerun()

    with col_btn_del:
        if st.button("🗑️ Remove", key=f"del_btn_{sym}", use_container_width=True):
            remove_from_watchlist(sym)
            if st.session_state.watchlist_view_sym == sym:
                st.session_state.watchlist_view_sym = None
            st.success(f"Removed **{sym}**")
            st.rerun()

    # Expand interactive chart if selected
    if st.session_state.watchlist_view_sym == sym:
        with st.container():
            st.info(f"Displaying interactive chart for **{sym}** with technical overlays:")
            try:
                hist_data = get_history_with_predictions(sym, days=120)
                pred_data = predict_next_close(sym)
                fig_stock = build_chart(
                    hist_data,
                    sym,
                    chart_type="candle",
                    rmse=pred_data["metrics"].get("rmse"),
                )
                st.plotly_chart(fig_stock, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load chart for {sym}: {e}")

    st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 6. DOWNLOAD REPORTS (PDF & CSV)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📥 Export & Download Watchlist Reports")
st.caption("Generate a PDF research document summarizing your monitored watchlist and portfolio valuation, or export as CSV.")

portfolio_sum = {
    "total_invested": total_invested,
    "current_value": total_current_val,
    "unrealized_pnl": total_current_val - total_invested,
    "pnl_pct": ((total_current_val - total_invested) / total_invested * 100.0) if total_invested > 0 else 0.0,
    "position_count": len([t for t in tracker_data if t["shares"] > 0]),
}

pdf_wl_bytes = generate_watchlist_pdf(tracker_data, portfolio_sum)
csv_str = pd.DataFrame(table_rows).to_csv(index=False)

col_exp_pdf, col_exp_csv = st.columns(2)

with col_exp_pdf:
    st.download_button(
        label="📄 Download Watchlist Report (PDF)",
        data=pdf_wl_bytes,
        file_name="watchlist_portfolio_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
        help="Download formatted PDF research document containing stock forecasts and portfolio summary.",
    )

with col_exp_csv:
    st.download_button(
        label="📊 Download Watchlist Data (CSV)",
        data=csv_str,
        file_name="watchlist_data.csv",
        mime="text/csv",
        use_container_width=True,
        help="Export all tracked prices, forecasts, and P&L metrics as a CSV spreadsheet.",
    )
