"""
pages/5_Backtest.py
--------------------
Backtest a simple ML-guided Long/Flat trading strategy:
  - If predicted next-day return > 0: hold the stock.
  - Else: stay in cash (0% return).

Features:
  - Configurable symbol, model choice (Linear Regression or Random Forest),
    evaluation period (6mo, 1y, 2y), and transaction cost (slippage + brokerage).
  - Normalized equity curve comparison: Strategy vs Buy-and-Hold benchmark.
  - Comprehensive performance metrics table (Total Return, Annualised Return / CAGR,
    Max Drawdown, Trades, Win Rate).
  - Clear, honest disclaimer on train/test data leakage for RELIANCE.NS and past performance.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_service.backtest import run_backtest
from ml_service.pdf_report import generate_backtest_pdf
from ml_service.risk_reward import render_risk_reward_widget

# --- Page config ---
st.set_page_config(
    page_title="Backtest Strategy — Stock Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Strategy Backtest")
st.caption(
    "Test a simple systematic trading rule powered by the ML model: "
    "**Long when predicted return > 0, Cash otherwise.** Evaluated without lookahead bias."
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# 1. SIDEBAR / CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Backtest Settings")

    symbol_input = st.text_input(
        "Stock Symbol (NSE)",
        value="RELIANCE.NS",
        help="Must be a valid Yahoo Finance ticker ending in .NS (e.g. RELIANCE.NS, TCS.NS, INFY.NS)",
    ).strip().upper()

    model_choice = st.selectbox(
        "Model",
        options=["Linear Regression (lr)", "Random Forest (rf)"],
        index=0,
        help="Choose which trained ML model generates next-day return predictions.",
    )
    model_key = "lr" if "lr" in model_choice else "rf"

    period_choice = st.selectbox(
        "Backtest Period",
        options=["6mo", "1y", "2y"],
        index=1,
        help="Historical lookback window from Yahoo Finance.",
    )

    cost_pct = st.number_input(
        "Transaction Cost per Trade (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.10,
        step=0.05,
        format="%.2f",
        help="Estimated brokerage, STT, and slippage charged on each position entry/exit. 0.10% = 10 bps.",
    )

    initial_capital = st.number_input(
        "Initial Capital (₹)",
        min_value=10000.0,
        max_value=10000000.0,
        value=100000.0,
        step=10000.0,
        format="%.0f",
        help="Starting portfolio cash in Rupees.",
    )

    run_btn = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

# Default or active state
if "backtest_data" not in st.session_state:
    st.session_state.backtest_data = None

# Trigger execution when clicked or on first load if empty
if run_btn or st.session_state.backtest_data is None:
    if not symbol_input.endswith(".NS"):
        st.error(f"❌ Invalid symbol '{symbol_input}'. Symbol must end with '.NS' (e.g. RELIANCE.NS).")
    else:
        with st.spinner(f"Running backtest for {symbol_input} using {model_choice}..."):
            try:
                bt_results = run_backtest(
                    symbol=symbol_input,
                    model_name=model_key,
                    period=period_choice,
                    transaction_cost=cost_pct / 100.0,
                    initial_capital=initial_capital,
                )
                st.session_state.backtest_data = bt_results
            except Exception as e:
                st.error(f"⚠️ Error running backtest: {e}")
                st.session_state.backtest_data = None

# ──────────────────────────────────────────────────────────────────────────────
# 2. RENDER RESULTS
# ──────────────────────────────────────────────────────────────────────────────
data = st.session_state.backtest_data

if data:
    s_m = data["strategy_metrics"]
    b_m = data["benchmark_metrics"]
    daily_df = data["daily_df"]

    # ── Summary banner ──
    st.subheader(f"📊 Results for {data['symbol']} ({data['start_date']} to {data['end_date']})")
    st.caption(
        f"Evaluated over **{data['num_days']} trading days** with **{data['transaction_cost_pct']:.2f}%** "
        f"transaction friction using **{model_choice.split(' ')[0]} {model_choice.split(' ')[1]}**."
    )

    # ── KPI cards ──
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric(
            "Strategy Return",
            f"{s_m['total_return']:+.2f}%",
            delta=f"{s_m['total_return'] - b_m['total_return']:+.2f}% vs B&H",
        )
    with k2:
        st.metric(
            "Buy & Hold Return",
            f"{b_m['total_return']:+.2f}%",
        )
    with k3:
        st.metric(
            "Strategy Max DD",
            f"{s_m['max_drawdown']:.2f}%",
            delta=f"{s_m['max_drawdown'] - b_m['max_drawdown']:+.2f}% vs B&H",
            delta_color="normal" if s_m['max_drawdown'] >= b_m['max_drawdown'] else "inverse",
        )
    with k4:
        st.metric(
            "Trades Executed",
            f"{s_m['trades']}",
            help="Total position changes (0->1 entries and 1->0 exits).",
        )
    with k5:
        st.metric(
            "Win Rate",
            f"{s_m['win_rate']:.1f}%",
            help="% of completed trades with positive net profit after transaction costs.",
        )

    st.markdown("")

    # ── Plotly Equity Curve ──
    fig_equity = go.Figure()

    fig_equity.add_trace(go.Scatter(
        x=daily_df["Date"],
        y=daily_df["Strategy_Equity"],
        mode="lines",
        name="ML Strategy (Long / Cash)",
        line=dict(color="#2563EB", width=2.5),
        hovertemplate="Strategy: ₹%{y:,.0f}<extra></extra>",
    ))

    fig_equity.add_trace(go.Scatter(
        x=daily_df["Date"],
        y=daily_df["Benchmark_Equity"],
        mode="lines",
        name="Buy & Hold Benchmark",
        line=dict(color="#64748B", width=1.8, dash="dot"),
        hovertemplate="Buy & Hold: ₹%{y:,.0f}<extra></extra>",
    ))

    fig_equity.update_layout(
        title=f"Equity Growth (Starting Capital: ₹{data['initial_capital']:,.0f})",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # ── Comparison Table ──
    st.subheader("📋 Performance Comparison Table")

    metrics_table = pd.DataFrame([
        {
            "Metric": "Total Return (%)",
            "ML Strategy": f"{s_m['total_return']:+.2f}%",
            "Buy & Hold": f"{b_m['total_return']:+.2f}%",
            "Difference": f"{s_m['total_return'] - b_m['total_return']:+.2f}%",
        },
        {
            "Metric": "Annualised Return / CAGR (%)",
            "ML Strategy": f"{s_m['annualised_return']:+.2f}%",
            "Buy & Hold": f"{b_m['annualised_return']:+.2f}%",
            "Difference": f"{s_m['annualised_return'] - b_m['annualised_return']:+.2f}%",
        },
        {
            "Metric": "Maximum Drawdown (%)",
            "ML Strategy": f"{s_m['max_drawdown']:.2f}%",
            "Buy & Hold": f"{b_m['max_drawdown']:.2f}%",
            "Difference": f"{s_m['max_drawdown'] - b_m['max_drawdown']:+.2f}%",
        },
        {
            "Metric": "Number of Trades (Rebalances)",
            "ML Strategy": f"{s_m['trades']}",
            "Buy & Hold": "1 (initial buy)",
            "Difference": f"{s_m['trades'] - 1:+d}",
        },
        {
            "Metric": "Completed Round-Trip Trades",
            "ML Strategy": f"{s_m['completed_trades']}",
            "Buy & Hold": "1",
            "Difference": f"{s_m['completed_trades'] - 1:+d}",
        },
        {
            "Metric": "Win Rate (%)",
            "ML Strategy": f"{s_m['win_rate']:.1f}%",
            "Buy & Hold": f"{b_m['win_rate']:.1f}% (up days)",
            "Difference": f"{s_m['win_rate'] - b_m['win_rate']:+.1f}%",
        },
        {
            "Metric": "Final Capital (₹)",
            "ML Strategy": f"₹{s_m['final_equity']:,.2f}",
            "Buy & Hold": f"₹{b_m['final_equity']:,.2f}",
            "Difference": f"₹{s_m['final_equity'] - b_m['final_equity']:+,.2f}",
        },
    ])

    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # ── Daily positions & drawdown expander ──
    with st.expander("🔍 View Daily Trade Signals & Drawdown Curve"):
        # Plot Drawdown
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Strategy_Drawdown"],
            mode="lines",
            name="Strategy Drawdown",
            line=dict(color="#EF4444", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
        ))
        fig_dd.add_trace(go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Benchmark_Drawdown"],
            mode="lines",
            name="Buy & Hold Drawdown",
            line=dict(color="#94A3B8", width=1.2, dash="dot"),
        ))
        fig_dd.update_layout(
            title="Drawdown (%) Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=280,
            hovermode="x unified",
            margin=dict(l=60, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Show trade log table if any
        if data["trade_events"]:
            st.markdown("**Completed Trade Log:**")
            trades_df = pd.DataFrame(data["trade_events"])
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")
            trades_df["entry_price"] = trades_df["entry_price"].apply(lambda v: f"₹{v:,.2f}")
            trades_df["exit_price"] = trades_df["exit_price"].apply(lambda v: f"₹{v:,.2f}")
            trades_df["trade_return"] = trades_df["trade_return"].apply(lambda v: f"{v*100:+.2f}%")
            trades_df["result"] = trades_df["is_win"].apply(lambda w: "🟢 Profit" if w else "🔴 Loss")
            trades_df = trades_df.rename(columns={
                "entry_date": "Entry Date",
                "exit_date": "Exit Date",
                "entry_price": "Entry Price",
                "exit_price": "Exit Price",
                "holding_days": "Holding Days",
                "trade_return": "Net Return",
                "result": "Outcome",
            })[["Entry Date", "Exit Date", "Entry Price", "Exit Price", "Holding Days", "Net Return", "Outcome"]]
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. RISK / REWARD & POSITION SIZING CALCULATOR
    # ──────────────────────────────────────────────────────────────────────────
    latest_close = float(daily_df["Close"].iloc[-1])
    last_pred_ret = float(daily_df["Predicted_Return"].iloc[-1])
    latest_pred_close = latest_close * (1.0 + last_pred_ret)
    daily_vol = float(daily_df["Asset_Return"].std()) * latest_close if len(daily_df) > 5 else latest_close * 0.02

    # Build lightweight chart_data list for swing levels from daily_df
    bt_chart_data = [
        {"low": float(row["Close"]) * 0.99, "high": float(row["Close"]) * 1.01}
        for _, row in daily_df.tail(20).iterrows()
    ]

    render_risk_reward_widget(
        current_price=latest_close,
        predicted_price=latest_pred_close,
        rmse=daily_vol,
        chart_data=bt_chart_data,
        key_prefix="bt_rr",
    )

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # 4. DOWNLOAD REPORTS & CSV EXPORT
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("📥 Export & Download Reports")
    st.caption("Download the comprehensive Backtest Performance Report as a PDF or export raw equity curve and trade logs as CSV.")

    pdf_bt_bytes = generate_backtest_pdf(
        symbol=data["symbol"],
        backtest_data=data,
    )
    daily_csv_str = daily_df.to_csv(index=False)

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        st.download_button(
            label="📄 Download Backtest PDF Report",
            data=pdf_bt_bytes,
            file_name=f"{data['symbol'].replace('.NS', '')}_backtest_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            help="Generates an executive PDF report containing performance comparison, trade log, and risk disclaimers.",
        )

    with col_dl2:
        st.download_button(
            label="📊 Download Daily Equity Curve (CSV)",
            data=daily_csv_str,
            file_name=f"{data['symbol'].replace('.NS', '')}_daily_equity.csv",
            mime="text/csv",
            use_container_width=True,
            help="Exports daily timestamps, positions, returns, and equity values for strategy and benchmark.",
        )

    with col_dl3:
        if data["trade_events"]:
            trades_csv_str = pd.DataFrame(data["trade_events"]).to_csv(index=False)
            st.download_button(
                label="📋 Download Completed Trades (CSV)",
                data=trades_csv_str,
                file_name=f"{data['symbol'].replace('.NS', '')}_trades_log.csv",
                mime="text/csv",
                use_container_width=True,
                help="Exports all completed entry and exit trade records with holding periods and net returns.",
            )
        else:
            st.button("📋 No Completed Trades", disabled=True, use_container_width=True)

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # 5. HONEST DISCLAIMERS & DATA LEAKAGE WARNING
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("⚠️ Critical Disclaimers & Methodology Notes")

    st.warning(
        "**1. Training / Testing Data Leakage Notice (RELIANCE.NS):**\n\n"
        "The model weights stored in `linear_regression_model_v3.pkl` and `random_forest_model_v3.pkl` "
        "were trained on historical data of **RELIANCE.NS** from 2022 to mid-2024. "
        "If you run a backtest on **RELIANCE.NS** covering that period, the model is evaluating on "
        "**in-sample data it has already seen during training** (lookahead/leakage contamination). "
        "To evaluate real out-of-sample performance, test on other NSE stocks (e.g. `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`) "
        "or time periods strictly after the training cutoff."
    )

    st.info(
        "**2. Past Performance is Not Indicative of Future Returns:**\n\n"
        "- **Transaction costs matter:** Frequent trading (switching between cash and long positions) "
        "accumulates transaction costs and exchange turnover fees, rapidly eating away marginal alpha.\n"
        "- **Execution assumptions:** This backtest assumes executions at daily closing prices with standard "
        "slippage. Real-world slippage on volatile market days can be significantly higher.\n"
        "- **Survivorship & regime changes:** Historical regimes with strong bull trends will show different "
        "long/flat dynamics compared to choppy sideways or bear markets."
    )
