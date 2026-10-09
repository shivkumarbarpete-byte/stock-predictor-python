"""
app.py
------
This is the HOME page of the Streamlit app.
Run the entire app with: streamlit run app.py

Streamlit automatically discovers any .py files inside the pages/ folder
and shows them as navigation links in the left sidebar.
"""

import streamlit as st

# --- Page configuration (must be the FIRST Streamlit call in the file) ---
st.set_page_config(
    page_title="Stock Market Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main home page content ---
st.title("📈 Stock Market Analysis & Prediction Tool")

st.markdown(
    """
    This tool uses a **Machine Learning model** (Linear Regression) trained on
    historical NSE (Indian stock market) data to predict the **next trading day's
    closing price** for any NIFTY 50 stock.

    It was built as a **pure Python** data science project using:
    `yfinance` · `scikit-learn` · `pandas` · `Streamlit` · `Plotly`
    """
)

st.divider()

# --- Quick feature overview using 3 columns ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Dashboard")
    st.write(
        "Search any NSE stock (e.g. `TCS.NS`), view the Actual vs Predicted "
        "price chart with SMA20 & SMA50 overlays, and see tomorrow's prediction."
    )

with col2:
    st.subheader("⭐ Watchlist")
    st.write(
        "Save your favourite stocks to a personal watchlist. "
        "Quickly view any saved stock's chart without re-typing the symbol."
    )

with col3:
    st.subheader("🤖 How the Model Works")
    st.write(
        "The model is trained using a time-ordered 80/20 train/test split "
        "on features: Close price, SMA20, SMA50, and EMA20."
    )

st.divider()

# --- Getting started note ---
st.info("👈 **Use the sidebar on the left to navigate between pages.**")

st.markdown(
    """
    #### Supported Stocks
    Any NSE stock with the `.NS` suffix works. Examples:
    `RELIANCE.NS` · `TCS.NS` · `INFY.NS` · `HDFCBANK.NS` · `WIPRO.NS` · `BAJFINANCE.NS`

    > **Note:** The model was trained on RELIANCE.NS data. Predictions for other stocks
    > use the same model weights, so treat them as approximate directional signals,
    > not exact prices.
    """
)
