"""
ml_service/risk_reward.py
-------------------------
Risk/Reward calculation module and position sizing calculator for Streamlit.

Provides:
  1. calculate_risk_reward(): Computes risk per share, reward per share, R:R ratio,
     and optional position sizing based on portfolio capital and max risk budget.
  2. suggest_trade_levels(): Automatically suggests Stop-Loss and Target prices
     derived from model prediction (+/- RMSE) and recent 20-day swing highs/lows.
  3. render_risk_reward_widget(): Clean, interactive Streamlit widget ready to be
     embedded in Dashboard or Backtest pages.
"""

from typing import Dict, List, Optional
import streamlit as st


def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    target: float,
    capital: Optional[float] = None,
    risk_budget_pct: float = 2.0,
) -> Dict:
    """
    Calculates detailed risk-to-reward dynamics and position sizing.

    Args:
        entry_price: Planned purchase price in Rupees.
        stop_loss: Planned protective exit price in Rupees.
        target: Planned profit-taking target in Rupees.
        capital: Total trading account capital in Rupees (optional).
        risk_budget_pct: Maximum portfolio loss tolerated on this trade (% of capital).

    Returns:
        Dict with metrics, risk-to-reward ratio, and recommended position size.
    """
    if entry_price <= 0:
        entry_price = 1.0
    if stop_loss <= 0:
        stop_loss = round(entry_price * 0.98, 2)
    if target <= 0:
        target = round(entry_price * 1.04, 2)

    risk_per_share = max(entry_price - stop_loss, 0.01)
    reward_per_share = max(target - entry_price, 0.0)

    risk_pct = (risk_per_share / entry_price) * 100.0
    reward_pct = (reward_per_share / entry_price) * 100.0

    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

    if rr_ratio >= 2.0:
        quality_label = "🟢 Favorable (≥ 1:2)"
        quality_badge = "success"
    elif rr_ratio >= 1.5:
        quality_label = "🟡 Acceptable (≥ 1:1.5)"
        quality_badge = "warning"
    else:
        quality_label = "🔴 Unfavorable (< 1:1.5)"
        quality_badge = "error"

    # Position sizing calculations
    position_info = {}
    if capital and capital > 0:
        max_rupee_risk = (capital * risk_budget_pct) / 100.0
        shares_by_risk = int(max_rupee_risk / risk_per_share) if risk_per_share > 0 else 0
        shares_by_capital = int(capital / entry_price)
        suggested_shares = max(min(shares_by_risk, shares_by_capital), 1)

        total_investment = round(suggested_shares * entry_price, 2)
        total_risk_rupees = round(suggested_shares * risk_per_share, 2)
        total_potential_profit = round(suggested_shares * reward_per_share, 2)

        position_info = {
            "max_allowed_risk": round(max_rupee_risk, 2),
            "suggested_shares": suggested_shares,
            "total_investment": total_investment,
            "total_risk_rupees": total_risk_rupees,
            "total_potential_profit": total_potential_profit,
        }

    return {
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "risk_per_share": round(risk_per_share, 2),
        "reward_per_share": round(reward_per_share, 2),
        "risk_pct": round(risk_pct, 2),
        "reward_pct": round(reward_pct, 2),
        "rr_ratio": round(rr_ratio, 2),
        "rr_ratio_formatted": f"1 : {rr_ratio:.2f}",
        "quality_label": quality_label,
        "quality_badge": quality_badge,
        "position_sizing": position_info,
    }


def suggest_trade_levels(
    current_price: float,
    predicted_price: float,
    rmse: float,
    chart_data: Optional[List[Dict]] = None,
) -> Dict:
    """
    Computes intelligent reference Stop-Loss and Target levels based on:
      1. Model Prediction & Volatility (RMSE)
      2. Recent 20-day Swing Lows & Highs
      3. Conservative 2% Risk / 1:2 Multiple Setup
    """
    current_price = max(current_price, 1.0)
    rmse = max(rmse, current_price * 0.01)

    # 1. Model-based Setup:
    # Target is the predicted close if higher than current, else 2.5% upside
    if predicted_price > current_price:
        model_target = round(predicted_price, 2)
    else:
        model_target = round(current_price * 1.03, 2)
    # Stop-loss at 1.5x model RMSE below current price (or at least 1.5% below)
    model_stop = round(max(current_price - (1.5 * rmse), current_price * 0.95), 2)

    # 2. Swing-based Setup (last 20 sessions):
    swing_low = current_price * 0.97
    swing_high = current_price * 1.04
    if chart_data and len(chart_data) >= 5:
        recent = chart_data[-20:]
        lows = [d["low"] for d in recent if d.get("low") is not None]
        highs = [d["high"] for d in recent if d.get("high") is not None]
        if lows:
            swing_low = min(lows)
        if highs:
            swing_high = max(highs)

    swing_stop = round(min(swing_low * 0.995, current_price * 0.99), 2)
    swing_target = round(max(swing_high * 1.005, current_price * 1.02), 2)

    # 3. Fixed 2% / 1:2 Setup:
    fixed_stop = round(current_price * 0.98, 2)
    fixed_target = round(current_price + (current_price - fixed_stop) * 2.0, 2)

    return {
        "model": {
            "name": "Model Prediction & RMSE",
            "stop_loss": model_stop,
            "target": model_target,
            "desc": f"Stop loss at entry - 1.5x RMSE (₹{rmse:,.2f}), Target at model prediction.",
        },
        "swing": {
            "name": "20-Day Swing High/Low",
            "stop_loss": swing_stop,
            "target": swing_target,
            "desc": f"Stop placed below 20-day swing support (₹{swing_low:,.2f}), target at recent resistance (₹{swing_high:,.2f}).",
        },
        "fixed": {
            "name": "Standard 2% Risk (1:2 R:R)",
            "stop_loss": fixed_stop,
            "target": fixed_target,
            "desc": "Protective 2% stop-loss with double reward target (4%).",
        },
    }


def render_risk_reward_widget(
    current_price: float,
    predicted_price: float,
    rmse: float,
    chart_data: Optional[List[Dict]] = None,
    key_prefix: str = "rr_widget",
) -> Dict:
    """
    Renders an interactive Streamlit card for calculating Risk/Reward and Position Sizing.
    Returns the computed dictionary so other features (e.g. PDF report) can read the values.
    """
    st.subheader("⚖️ Risk / Reward & Trade Setup Calculator")
    st.caption(
        "Plan your trade entry, stop-loss, and profit targets. Levels can be auto-populated "
        "from the ML model prediction or recent swing highs/lows."
    )

    suggestions = suggest_trade_levels(current_price, predicted_price, rmse, chart_data)

    # Presets row
    col_preset, col_cap = st.columns([3, 2])

    with col_preset:
        preset_choice = st.radio(
            "Auto-Preset Strategy",
            options=["Model & RMSE", "20-Day Swing Levels", "Standard 2% (1:2)", "Custom Manual"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_preset",
            help="Select an automated trade setup preset or customize prices manually below.",
        )

    with col_cap:
        portfolio_cap = st.number_input(
            "Trading Capital (₹)",
            min_value=5000.0,
            max_value=10000000.0,
            value=100000.0,
            step=10000.0,
            format="%.0f",
            key=f"{key_prefix}_capital",
            help="Used to calculate position sizing (recommended number of shares).",
        )

    # Determine default values based on preset
    if preset_choice == "Model & RMSE":
        def_entry = current_price
        def_stop = suggestions["model"]["stop_loss"]
        def_target = suggestions["model"]["target"]
        st.info(f"💡 **Active Preset:** {suggestions['model']['desc']}")
    elif preset_choice == "20-Day Swing Levels":
        def_entry = current_price
        def_stop = suggestions["swing"]["stop_loss"]
        def_target = suggestions["swing"]["target"]
        st.info(f"💡 **Active Preset:** {suggestions['swing']['desc']}")
    elif preset_choice == "Standard 2% (1:2)":
        def_entry = current_price
        def_stop = suggestions["fixed"]["stop_loss"]
        def_target = suggestions["fixed"]["target"]
        st.info(f"💡 **Active Preset:** {suggestions['fixed']['desc']}")
    else:
        def_entry = current_price
        def_stop = round(current_price * 0.98, 2)
        def_target = round(current_price * 1.04, 2)

    # Input Fields
    col_in1, col_in2, col_in3, col_in4 = st.columns(4)

    with col_in1:
        entry_val = st.number_input(
            "Entry Price (₹)",
            min_value=0.1,
            value=float(def_entry),
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_entry",
        )

    with col_in2:
        stop_val = st.number_input(
            "Stop-Loss (₹)",
            min_value=0.05,
            value=float(def_stop),
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_stop",
            help="Exit price if trade moves against you.",
        )

    with col_in3:
        target_val = st.number_input(
            "Target Price (₹)",
            min_value=0.1,
            value=float(def_target),
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_target",
            help="Profit-taking exit price.",
        )

    with col_in4:
        risk_tol_pct = st.number_input(
            "Risk Tolerance (%)",
            min_value=0.25,
            max_value=10.0,
            value=2.0,
            step=0.25,
            format="%.2f",
            key=f"{key_prefix}_risk_pct",
            help="Maximum portfolio risk allocated to this trade (default: 2%).",
        )

    # Calculate
    calc = calculate_risk_reward(
        entry_price=entry_val,
        stop_loss=stop_val,
        target=target_val,
        capital=portfolio_cap,
        risk_budget_pct=risk_tol_pct,
    )

    # Output Metric Cards
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            label="🛑 Risk / Share",
            value=f"₹{calc['risk_per_share']:,.2f}",
            delta=f"-{calc['risk_pct']:.2f}% (Stop Loss)",
            delta_color="inverse",
        )

    with m2:
        st.metric(
            label="🎯 Reward / Share",
            value=f"₹{calc['reward_per_share']:,.2f}",
            delta=f"+{calc['reward_pct']:.2f}% (Target)",
            delta_color="normal",
        )

    with m3:
        st.metric(
            label="⚖️ Risk : Reward Ratio",
            value=calc["rr_ratio_formatted"],
            help="A ratio of at least 1:2 is generally recommended for positive long-term expectancy.",
        )

    with m4:
        st.metric(
            label="Trade Setup Quality",
            value=calc["quality_label"],
        )

    # Position sizing summary box
    pos = calc.get("position_sizing", {})
    if pos:
        st.markdown(
            f"""
            <div style="background-color: rgba(37, 99, 235, 0.08); border: 1px solid rgba(37, 99, 235, 0.25);
                        border-radius: 8px; padding: 12px 18px; margin-top: 10px;">
                <div style="font-weight: 600; color: #1E40AF; margin-bottom: 6px;">
                    💼 Recommended Position Sizing (Account Capital: ₹{portfolio_cap:,.0f} · Max Risk {risk_tol_pct}% = ₹{pos['max_allowed_risk']:,.2f})
                </div>
                <div style="font-size: 0.95rem; color: #1F2937;">
                    • <strong>Buy Quantity:</strong> <strong>{pos['suggested_shares']} shares</strong>
                    (Total capital required: <strong>₹{pos['total_investment']:,.2f}</strong>)<br>
                    • <strong>Capital at Risk:</strong> <strong>₹{pos['total_risk_rupees']:,.2f}</strong> if Stop-Loss is hit<br>
                    • <strong>Potential Profit:</strong> <strong>₹{pos['total_potential_profit']:,.2f}</strong> if Target is reached
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return calc
