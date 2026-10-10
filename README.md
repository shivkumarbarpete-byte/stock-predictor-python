# 📈 Stock Market Analysis & Prediction Tool

A **pure Python** data science project that fetches live NSE (Indian stock market) data,
computes technical indicators, trains ML models, and displays an interactive dashboard.

Built as a portfolio project for data science / ML interviews.

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-predictor-python-juipuglwqubpvlatzptjbh.streamlit.app/)

**👉 [Open the app](https://stock-predictor-python-juipuglwqubpvlatzptjbh.streamlit.app/)**

Hosted on Streamlit Community Cloud. If the app has been idle, it may take a few seconds to wake up.

---

## ✨ Features

| Feature | Details |
|---|---|
| Live data | Yahoo Finance via `yfinance` — any NSE stock (`.NS` suffix) |
| Technical indicators | SMA20, SMA50, EMA20, RSI (14-day) |
| ML models | Linear Regression and Random Forest — trained and compared side by side |
| Naive baseline | Always compared against "tomorrow = today's close" |
| Metrics | RMSE, MAE, R² on a held-out time-ordered test set |
| Prediction range | Predicted price ± RMSE (honest uncertainty estimate) |
| Interactive UI | Streamlit multi-page app with Plotly charts |
| Watchlist | Save favourite stocks to a local JSON file |
| Model info page | Feature importance chart, comparison table, honest disclaimer |

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Charts | [Plotly](https://plotly.com/python/) |
| Data | [yfinance](https://github.com/ranaroussi/yfinance), [pandas](https://pandas.pydata.org/) |
| ML | [scikit-learn](https://scikit-learn.org/) |
| Model storage | [joblib](https://joblib.readthedocs.io/) |
| Numerics | [numpy](https://numpy.org/) |

---

## 📁 Folder Structure

```
stock-predictor-python/
│
├── app.py                          # Streamlit home page (run this to start)
│
├── ml_service/                     # All ML logic — fetch, indicators, train, predict
│   ├── config.py                   # SINGLE SOURCE OF TRUTH: features list + file paths
│   ├── stock_utils.py              # Core pipeline: fetch → indicators → predict
│   ├── train_model.py              # Training script: Naive / LR / RF comparison
│   ├── compute_indicators.py       # SMA20, SMA50, EMA20, RSI calculations
│   ├── fetch_data.py               # yfinance download wrapper
│   ├── linear_regression_model.pkl # Original LR model (4 features, kept as backup)
│   ├── linear_regression_model_v2.pkl  # New LR model (5 features, with RSI)
│   ├── random_forest_model.pkl     # Random Forest model (5 features, with RSI)
│   └── metrics.json                # RMSE/MAE/R² for all models (written by train_model.py)
│
├── pages/                          # Streamlit auto-discovers these as sidebar pages
│   ├── 1_Dashboard.py              # Stock search, chart, prediction, watchlist button
│   ├── 2_Watchlist.py              # Saved stocks with inline charts
│   └── 3_Model_Info.py             # Model comparison table + feature importance chart
│
├── utils/
│   ├── watchlist.py                # load_watchlist() / save_watchlist() using JSON
│   └── chart_helpers.py            # build_chart() — Plotly 4-line chart builder
│
├── data/
│   └── watchlist.json              # Persisted list of saved stock symbols
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── INTERVIEW_NOTES.md              # 12 likely interview Q&A about this project
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.9 or higher
- Internet connection (for live stock data from Yahoo Finance)

### Step-by-step

```bash
# 1. Navigate into the project folder
cd "stock-predictor-python"

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy the original model file into the ml_service folder (first time only)
Copy-Item "ml-service\linear_regression_model.pkl" "ml_service\linear_regression_model.pkl"

# 6. Train the new models (LR v2 + Random Forest) and generate metrics.json
python ml_service/train_model.py

# 7. Launch the Streamlit app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

> **Note:** Step 6 fetches ~2 years of RELIANCE.NS data, trains two models,
> and saves `linear_regression_model_v2.pkl`, `random_forest_model.pkl`, and `metrics.json`
> inside `ml_service/`. This takes ~30–60 seconds on first run.

---

## 📸 Screenshots

> _[Add a screenshot of the Dashboard page here]_

> _[Add a screenshot of the Model Info page here]_

---

## 📝 Key Design Decisions

### Why time-ordered train/test split (no shuffling)?
Stock data is sequential. Shuffling would let the model see "future" prices during
training — a classic mistake called **data leakage**. We always use the oldest 80%
for training and the newest 20% for testing.

### Why a naive baseline?
Before claiming an ML model is "good", you must prove it beats the simplest possible guess.
Here, the naive baseline is: "tomorrow's price = today's close". If our ML model
barely beats this, it hasn't learned anything useful.

### Why three metrics?
- **RMSE** — penalises large errors more heavily (important in finance)
- **MAE** — average error in plain rupees (easy to explain to anyone)
- **R²** — shows what % of price variation the model can explain

### Why RSI as a feature?
RSI (Relative Strength Index) is a momentum indicator widely used in technical analysis.
Adding it means the model has information about whether a stock is potentially overbought
or oversold — a domain-knowledge feature that goes beyond pure price history.

---

## ⚠️ Disclaimer

This tool is for **learning and portfolio demonstration only**.
It is not financial advice and should not be used to make real investment decisions.