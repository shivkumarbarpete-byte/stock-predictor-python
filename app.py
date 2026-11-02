"""
app.py
------
Home page of the NSE Stock Predictor & Portfolio Analytics Suite.
Run with: streamlit run app.py
"""

import streamlit as st

# --- Page configuration ---
st.set_page_config(
    page_title="NSE Stock Predictor & Risk Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 NSE Stock Predictor & Quantitative Analytics Suite")
st.caption("AI-Guided Stock Forecasts · Technical Indicators · Systematic Backtesting · Risk Management")

st.markdown(
    """
    An institutional-grade **data science and algorithmic trading workspace** built in pure Python.
    Features **scale-free Machine Learning architectures** (Linear Regression, Random Forest, XGBoost, LightGBM)
    trained on historical NSE equities, integrated with **live financial news sentiment analysis** and
    interactive **Plotly visualization**.
    """
)

st.divider()

# --- Feature Showcase Grid ---
st.subheader("🚀 Platform Modules")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; height: 100%;">
            <h4 style="color: #2563EB; margin-top: 0;">📊 1. Stock Dashboard</h4>
            <p style="font-size: 0.9rem; color: #475569;">
                Search any NSE ticker (e.g. <code>RELIANCE.NS</code>), dynamically switch between 4 ML models,
                inspect interactive Candlestick & Plotly charts, live news sentiment, and trade setup calculators.
            </p>
            <p style="font-size: 0.85rem; color: #64748B;">• Interactive Plotly with range buttons<br>• Live VADER Sentiment & AI Summaries<br>• PDF & CSV Research Export</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; height: 100%;">
            <h4 style="color: #D97706; margin-top: 0;">⭐ 2. Watchlist & Portfolio</h4>
            <p style="font-size: 0.9rem; color: #475569;">
                Track your personal watchlist of monitored stocks with real-time model predictions, RSI indicators,
                trend signals, and live portfolio valuation (invested capital & unrealized P&L).
            </p>
            <p style="font-size: 0.85rem; color: #64748B;">• Direct ticker add with auto-formatting<br>• Multi-stock tracker table<br>• Dedicated Watchlist PDF Report</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; height: 100%;">
            <h4 style="color: #16A34A; margin-top: 0;">🤖 3. Model Performance</h4>
            <p style="font-size: 0.9rem; color: #475569;">
                Compare 5 models against the Naive Baseline. View Directional Accuracy with the 50% coin-flip reference,
                5-fold Walk-Forward Validation, and tree feature importance scores.
            </p>
            <p style="font-size: 0.85rem; color: #64748B;">• Hyperparameter Tuning results<br>• Walk-Forward Expanding Windows<br>• Honest Data Leakage Disclaimers</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
c4, c5 = st.columns(2)

with c4:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; height: 100%;">
            <h4 style="color: #9333EA; margin-top: 0;">⚖️ 4. Multi-Stock Compare</h4>
            <p style="font-size: 0.9rem; color: #475569;">
                Compare 2 to 4 NSE stocks normalized to percentage return from day zero over the last 120 sessions.
                Evaluate cross-asset drawdowns and annualized volatilities side-by-side.
            </p>
            <p style="font-size: 0.85rem; color: #64748B;">• Normalized zero-base lines<br>• Risk & volatility tables<br>• Instant CSV table download</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; height: 100%;">
            <h4 style="color: #0284C7; margin-top: 0;">📈 5. Strategy Backtesting</h4>
            <p style="font-size: 0.9rem; color: #475569;">
                Simulate systematic long/flat rules powered by machine learning forecasts with configurable
                dynamic Stop-Loss, Take-Profit targets, and real-world transaction friction.
            </p>
            <p style="font-size: 0.85rem; color: #64748B;">• Sharpe & Sortino Ratios (Rf = 6%)<br>• Maximum Drawdown & Win/Loss Ratios<br>• Interactive Trade Log with exit reasons</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# --- Quick guidance ---
col_guide, col_stocks = st.columns([1.5, 2])

with col_guide:
    st.subheader("🧭 Getting Started")
    st.markdown(
        """
        1. Select **📊 Dashboard** in the left sidebar to analyze any NSE equity.
        2. Test multiple models (**Random Forest**, **XGBoost**, **LightGBM**, **Linear Regression**).
        3. Save favorites to **⭐ Watchlist** and track your portfolio value.
        4. Run strategy simulations in **📈 Backtest** with protective stop-loss rules.
        """
    )

with col_stocks:
    st.subheader("🏷️ Popular Supported Tickers")
    st.markdown(
        """
        Any NSE ticker ending in `.NS` works out-of-the-box (omitting `.NS` will auto-correct):
        - `RELIANCE.NS` (Energy / Conglomerate)
        - `TCS.NS` & `INFY.NS` (Information Technology)
        - `HDFCBANK.NS` & `ICICIBANK.NS` (Banking / Financials)
        - `TATAMOTORS.NS` (Automotive)
        - `BHARTIARTL.NS` (Telecom)
        """
    )
