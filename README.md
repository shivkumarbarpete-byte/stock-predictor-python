# Stock Market Analysis & Prediction Tool

A pure-Python data science project that predicts the **next trading day's closing price** for NSE (Indian market) stocks using machine learning on historical data.

## 🚀 Live Demo

**👉 [Open the app](https://stock-predictor-python-juipuglwqubpvlatzptjbh.streamlit.app/)**

Hosted on Streamlit Community Cloud. If the app has been idle, it may take a few seconds to wake up.

## ✨ Features

- **Dashboard**: search any NSE stock (e.g. `TCS.NS`), view the Actual vs Predicted chart with SMA20 and SMA50 overlays, and see the next-day prediction.
- **Watchlist**: save favourite stocks and open their charts without re-typing the symbol.
- **Model Info**: see how the model was trained and how it compares against a naive baseline (metrics are read from `ml_service/metrics.json`).

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| UI | Streamlit |
| Charts | Plotly |
| Data | yfinance, pandas, numpy |
| ML | scikit-learn (Linear Regression, Random Forest), joblib |

## 🤖 How the Model Works

1. Historical daily data is downloaded with `yfinance`.
2. Technical indicators are computed as features (SMA20, SMA50, EMA20, RSI).
3. Target: the **next day's Close price**.
4. **Time-ordered 80/20 train/test split** (no shuffling, so the model never trains on the future).
5. Models are compared against a **naive baseline** (tomorrow = today) using the same test period:
   - Naive baseline
   - Linear Regression
   - Random Forest
6. Metrics are saved to `ml_service/metrics.json` and shown on the Model Info page.

## ⚠️ Limitations

- Stock prices are very hard to predict. This is a learning project, **not financial advice**.
- The model was trained on `RELIANCE.NS` data. Predictions for other stocks reuse the same model, so treat them as approximate directional signals, not exact prices.
- The watchlist is stored in `data/watchlist.json`. On Streamlit Cloud the file system is not permanent, so the watchlist can reset when the app restarts.
- Live data comes from Yahoo Finance and can occasionally be slow or rate-limited.

## 📁 Project Structure

```
stock-predictor-python/
├── app.py                 # Home page
├── pages/
│   ├── 1_Dashboard.py     # Search + chart + prediction
│   ├── 2_Watchlist.py     # Saved stocks
│   └── 3_Model_Info.py    # Model details and metrics
├── ml_service/            # Training code, saved model, metrics.json
├── utils/
│   ├── watchlist.py       # Watchlist read/write helpers
│   └── chart_helpers.py   # Plotly chart builders
├── data/
│   └── watchlist.json
├── requirements.txt
├── README.md
└── INTERVIEW_NOTES.md
```

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/shivkumarbarpete-byte/stock-predictor-python.git
cd stock-predictor-python

# 2. Create and activate a virtual environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## 🌐 Deployment Notes

- Deployed on Streamlit Community Cloud with **Python 3.11**.
- `requirements.txt` pins `pandas`, `scikit-learn` and `joblib`, and keeps `numpy<2`, so the saved model loads with the same library versions it was trained with.

## 👤 Author

**Shiv Kumar Barpete**
GitHub: [shivkumarbarpete-byte](https://github.com/shivkumarbarpete-byte)

---

*Related: a MERN-stack version of this project (React + Express + MongoDB + FastAPI ML service) is maintained separately.*