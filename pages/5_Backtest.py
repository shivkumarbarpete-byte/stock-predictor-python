"""
pages/5_Backtest.py
--------------------
Backtest ML-guided trading strategies with dynamic Stop-Loss and Take-Profit rules.

Features:
  - Configurable symbol, model choice (Linear Regression, Random Forest, XGBoost, LightGBM),
    evaluation period (6mo, 1y, 2y), and transaction cost (brokerage & slippage).
  - Stop-Loss & Take-Profit targets: protects downside and locks in gains dynamically.
  - Advanced Trading Metrics:
      * Sharpe Ratio (annualized vs 6% risk-free rate)
      * Sortino Ratio (downside risk deviation)
      * Maximum Drawdown (peak-to-trough drop)
      * Win/Loss Ratio & Profit Factor
  - Normalized equity curve comparison: Strategy vs Buy-and-Hold benchmark.
  - Detailed Trade Log with Exit Triggers (Stop Loss, Take Profit, Signal Flip).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_service.backtest import run_backtest
from ml_service.stock_utils import get_available_models
from ml_service.config import MODEL_DISPLAY_NAMES
from ml_service.pdf_report import generate_backtest_pdf
from ml_service.risk_reward import render_risk_reward_widget

# --- Page config ---
st.set_page_config(
    page_title="Backtest Strategy — Stock Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Strategy Backtest & Risk Management")
st.caption(
    "Simulate systematic ML trading rules with protective Stop-Loss and Take-Profit mechanisms. "
    "Evaluated chronologically without lookahead bias."
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

    # Dynamic model options
    avail_models = get_available_models()
    if not avail_models:
        avail_models = ["lr", "rf", "xgb", "lgbm"]

    model_labels = [f"{MODEL_DISPLAY_NAMES.get(k, k)} ({k})" for k in avail_models]
    model_choice = st.selectbox(
        "Model",
        options=model_labels,
        index=0,
        help="Choose which trained ML model generates next-day return predictions.",
    )
    # Extract short key (e.g. 'lr', 'rf', 'xgb', 'lgbm')
    model_key = model_choice.split("(")[-1].replace(")", "").strip()

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

    st.markdown("---")
    st.subheader("🛡️ Risk Rules (SL / TP)")

    enable_sl = st.checkbox("Enable Stop-Loss", value=True, help="Exit trade if price drops below entry by X%")
    if enable_sl:
        sl_input = st.slider(
            "Stop-Loss Threshold (%)",
            min_value=0.5,
            max_value=15.0,
            value=3.0,
            step=0.5,
            help="Maximum acceptable loss per trade from entry price.",
        )
        stop_loss_val = sl_input / 100.0
    else:
        stop_loss_val = None

    enable_tp = st.checkbox("Enable Take-Profit", value=True, help="Exit trade when price reaches target of Y%")
    if enable_tp:
        tp_input = st.slider(
            "Take-Profit Target (%)",
            min_value=1.0,
            max_value=30.0,
            value=6.0,
            step=0.5,
            help="Target profit gain per trade from entry price.",
        )
        take_profit_val = tp_input / 100.0
    else:
        take_profit_val = None

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
                    stop_loss_pct=stop_loss_val,
                    take_profit_pct=take_profit_val,
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
    
    sl_badge = f"🛑 Stop-Loss: **{data['stop_loss_pct']:.1f}%**" if data['stop_loss_pct'] else "🛑 Stop-Loss: *Disabled*"
    tp_badge = f"🎯 Take-Profit: **{data['take_profit_pct']:.1f}%**" if data['take_profit_pct'] else "🎯 Take-Profit: *Disabled*"
    model_disp = MODEL_DISPLAY_NAMES.get(data['model_name'], data['model_name'])

    st.caption(
        f"Evaluated over **{data['num_days']} trading days** with **{data['transaction_cost_pct']:.2f}%** friction "
        f"using **{model_disp}**. Configuration: {sl_badge} · {tp_badge}"
    )

    # ── Primary KPI Cards ──
    st.markdown("#### 🏆 Core Performance Highlights")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric(
            "Strategy Return",
            f"{s_m['total_return']:+.2f}%",
            delta=f"{s_m['total_return'] - b_m['total_return']:+.2f}% vs B&H",
        )
    with k2:
        st.metric(
            "Annualized CAGR",
            f"{s_m['annualised_return']:+.2f}%",
            delta=f"{s_m['annualised_return'] - b_m['annualised_return']:+.2f}% vs B&H",
        )
    with k3:
        st.metric(
            "Buy & Hold Return",
            f"{b_m['total_return']:+.2f}%",
            delta=f"CAGR: {b_m['annualised_return']:+.2f}%",
            delta_color="off",
        )
    with k4:
        st.metric(
            "Win Rate",
            f"{s_m['win_rate']:.1f}%",
            delta=f"{s_m['win_count']} Wins / {s_m['loss_count']} Losses",
            delta_color="off",
            help="% of completed trades with positive net profit after transaction costs.",
        )
    with k5:
        st.metric(
            "Ending Portfolio",
            f"₹{s_m['final_equity']:,.2f}",
            delta=f"₹{s_m['final_equity'] - data['initial_capital']:+,.2f} P&L",
        )

    # ── Advanced Institutional Trading Metrics ──
    st.markdown("#### 🔬 Institutional Risk & Trading Metrics")
    r1, r2, r3, r4, r5 = st.columns(5)

    with r1:
        st.metric(
            "Sharpe Ratio (Rf=6%)",
            f"{s_m['sharpe_ratio']:.2f}",
            delta=f"{s_m['sharpe_ratio'] - b_m['sharpe_ratio']:+.2f} vs B&H ({b_m['sharpe_ratio']:.2f})",
            help=(
                "Annualized risk-adjusted excess return per unit of total risk. "
                "Higher is better (>1.0 is considered good, >2.0 is very good)."
            ),
        )

    with r2:
        st.metric(
            "Sortino Ratio",
            f"{s_m['sortino_ratio']:.2f}",
            delta=f"{s_m['sortino_ratio'] - b_m['sortino_ratio']:+.2f} vs B&H ({b_m['sortino_ratio']:.2f})",
            help=(
                "Focuses strictly on harmful downside risk instead of total volatility. "
                "Only penalizes negative excess volatility."
            ),
        )

    with r3:
        st.metric(
            "Maximum Drawdown",
            f"{s_m['max_drawdown']:.2f}%",
            delta=f"{s_m['max_drawdown'] - b_m['max_drawdown']:+.2f}% vs B&H ({b_m['max_drawdown']:.2f}%)",
            delta_color="normal" if s_m['max_drawdown'] >= b_m['max_drawdown'] else "inverse",
            help="Largest peak-to-trough portfolio decline. Closer to 0% is safer.",
        )

    with r4:
        st.metric(
            "Win / Loss Ratio",
            f"{s_m['win_loss_ratio']:.2f}x",
            delta=f"Avg Win: {s_m['avg_win_pct']:+.1f}% | Avg Loss: -{s_m['avg_loss_pct']:.1f}%",
            delta_color="off",
            help="Average profit on winning trades divided by average loss on losing trades. >1.0 means winners outweigh losers.",
        )

    with r5:
        st.metric(
            "Profit Factor",
            f"{s_m['profit_factor']:.2f}",
            delta="Gross Profit / Gross Loss",
            delta_color="off",
            help="Total gross profits divided by total gross losses. Values above 1.5 indicate robust edge.",
        )

    st.markdown("")

    # ── Plotly Equity Curve ──
    fig_equity = go.Figure()

    fig_equity.add_trace(go.Scatter(
        x=daily_df["Date"],
        y=daily_df["Strategy_Equity"],
        mode="lines",
        name=f"ML Strategy ({model_disp})",
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
    st.subheader("📋 Comprehensive Performance & Risk Table")

    metrics_table = pd.DataFrame([
        {
            "Metric": "Total Net Return (%)",
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
            "Metric": "Sharpe Ratio (Rf = 6%)",
            "ML Strategy": f"{s_m['sharpe_ratio']:.2f}",
            "Buy & Hold": f"{b_m['sharpe_ratio']:.2f}",
            "Difference": f"{s_m['sharpe_ratio'] - b_m['sharpe_ratio']:+.2f}",
        },
        {
            "Metric": "Sortino Ratio (Downside Deviation)",
            "ML Strategy": f"{s_m['sortino_ratio']:.2f}",
            "Buy & Hold": f"{b_m['sortino_ratio']:.2f}",
            "Difference": f"{s_m['sortino_ratio'] - b_m['sortino_ratio']:+.2f}",
        },
        {
            "Metric": "Maximum Drawdown (%)",
            "ML Strategy": f"{s_m['max_drawdown']:.2f}%",
            "Buy & Hold": f"{b_m['max_drawdown']:.2f}%",
            "Difference": f"{s_m['max_drawdown'] - b_m['max_drawdown']:+.2f}%",
        },
        {
            "Metric": "Profit Factor (Gross Win / Gross Loss)",
            "ML Strategy": f"{s_m['profit_factor']:.2f}",
            "Buy & Hold": f"{b_m['profit_factor']:.2f}",
            "Difference": f"{s_m['profit_factor'] - b_m['profit_factor']:+.2f}",
        },
        {
            "Metric": "Win / Loss Ratio (Avg Win / Avg Loss)",
            "ML Strategy": f"{s_m['win_loss_ratio']:.2f}x ({s_m['avg_win_pct']:+.1f}% / -{s_m['avg_loss_pct']:.1f}%)",
            "Buy & Hold": "—",
            "Difference": "—",
        },
        {
            "Metric": "Trade Win Rate (%)",
            "ML Strategy": f"{s_m['win_rate']:.1f}% ({s_m['win_count']}W / {s_m['loss_count']}L)",
            "Buy & Hold": f"{b_m['win_rate']:.1f}% (up days)",
            "Difference": f"{s_m['win_rate'] - b_m['win_rate']:+.1f}%",
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
            "Metric": "Final Capital (₹)",
            "ML Strategy": f"₹{s_m['final_equity']:,.2f}",
            "Buy & Hold": f"₹{b_m['final_equity']:,.2f}",
            "Difference": f"₹{s_m['final_equity'] - b_m['final_equity']:+,.2f}",
        },
    ])

    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # ── Daily positions & drawdown expander ──
    with st.expander("🔍 View Daily Trade Signals & Drawdown Curve"):
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

        # Show trade log table with exit reasons
        if data["trade_events"]:
            st.markdown("**Completed Trade Log:**")
            trades_df = pd.DataFrame(data["trade_events"])
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")
            trades_df["entry_price"] = trades_df["entry_price"].apply(lambda v: f"₹{v:,.2f}")
            trades_df["exit_price"] = trades_df["exit_price"].apply(lambda v: f"₹{v:,.2f}")
            trades_df["trade_return"] = trades_df["trade_return"].apply(lambda v: f"{v*100:+.2f}%")
            trades_df["result"] = trades_df["is_win"].apply(lambda w: "🟢 Profit" if w else "🔴 Loss")

            # Format exit trigger with icons
            def _format_exit_reason(reason):
                if reason == "Stop Loss":
                    return "🛑 Stop Loss Hit"
                elif reason == "Take Profit":
                    return "🎯 Take Profit Hit"
                elif reason == "Signal Flip":
                    return "🔄 Signal Flip (Model)"
                return "🏁 End of Period"

            trades_df["Exit Trigger"] = trades_df.get("exit_reason", "Signal Flip").apply(_format_exit_reason)

            trades_display = trades_df.rename(columns={
                "entry_date": "Entry Date",
                "exit_date": "Exit Date",
                "entry_price": "Entry Price",
                "exit_price": "Exit Price",
                "holding_days": "Holding Days",
                "trade_return": "Net Return",
                "result": "Outcome",
            })[["Entry Date", "Exit Date", "Entry Price", "Exit Price", "Holding Days", "Net Return", "Outcome", "Exit Trigger"]]

            st.dataframe(trades_display, use_container_width=True, hide_index=True)

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. RISK / REWARD & POSITION SIZING CALCULATOR
    # ──────────────────────────────────────────────────────────────────────────
    latest_close = float(daily_df["Close"].iloc[-1])
    last_pred_ret = float(daily_df["Predicted_Return"].iloc[-1])
    latest_pred_close = latest_close * (1.0 + last_pred_ret)
    daily_vol = float(daily_df["Asset_Return"].std()) * latest_close if len(daily_df) > 5 else latest_close * 0.02

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
        "The model weights stored in the `.pkl` files were trained on historical data of **RELIANCE.NS** from 2022 to mid-2024. "
        "If you run a backtest on **RELIANCE.NS** covering that period, the model is evaluating on "
        "**in-sample data it has already seen during training** (lookahead/leakage contamination). "
        "To evaluate real out-of-sample performance, test on other NSE stocks (e.g. `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`) "
        "or time periods strictly after the training cutoff."
    )

    st.info(
        "**2. Past Performance is Not Indicative of Future Returns:**\n\n"
        "- **Transaction costs matter:** Frequent trading (switching between cash and long positions) "
        "accumulates transaction costs and exchange turnover fees, rapidly eating away marginal alpha.\n"
        "- **Stop-Loss / Take-Profit Slippage:** Real-world gap openings can cause executions below the stop-loss price. "
        "Intraday prices are checked against daily high and low extremes.\n"
        "- **Survivorship & regime changes:** Historical regimes with strong bull trends will show different "
        "long/flat dynamics compared to choppy sideways or bear markets."
    )
