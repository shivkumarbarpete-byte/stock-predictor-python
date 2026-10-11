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
| Prediction target | **Next-day return** (not absolute price), converted to price as `Close × (1 + predicted return)` |
| Scale-free features | SMA20/Close, SMA50/Close, EMA20/Close, RSI — one model works across stocks of any price level |
| ML models | Linear Regression and Random Forest — trained and compared side by side |
| Naive baseline | Always compared against "tomorrow = today's close" |
| Metrics | RMSE, MAE, R² on a held-out time-ordered test set |
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

## 📊 Model Results

Models are trained on about 2 years of `RELIANCE.NS` data with a time-ordered split
(oldest 80% for training, newest 20% for testing). The target is the **next-day return**.

| Model | R² (on next-day return) |
|---|---|
| Naive baseline (return = 0) | -0.008 |
| Linear Regression | -0.036 |
| Random Forest | -0.171 |

**What this means:** neither model beats the naive baseline. Daily stock returns are
very close to random, so a simple model using only price-based indicators cannot
predict them reliably. The predicted-price line on the dashboard looks close to the
actual line mostly because tomorrow's price is usually close to today's price.
This project reports that honestly instead of hiding it.

---

## 📁 Folder Structure

```
stock-predictor-python/
│
├── app.py                          # Streamlit home page (run this to start)
│
├── ml_service/                     # All ML logic — fetch, indicators, train, predict
│   ├── config.py                   # SINGLE SOURCE OF TRUTH: features list + file paths
│   ├── features.py                 # Scale-free feature builder (ratios + RSI)
│   ├── stock_utils.py              # Core pipeline: fetch → features → predict
│   ├── train_model.py              # Training script: Naive / LR / RF comparison
│   ├── compute_indicators.py       # SMA20, SMA50, EMA20, RSI calculations
│   ├── fetch_data.py               # yfinance download wrapper
│   ├── *.pkl                       # Trained models (return-based)
│   └── metrics.json                # RMSE/MAE/R² for all models (written by train_model.py)
│
├── pages/                          # Streamlit auto-discovers these as sidebar pages
│   ├── 1_Dashboard.py              # Stock search, chart, prediction, watchlist button
│   ├── 2_Watchlist.py              # Saved stocks with inline charts
│   └── 3_Model_Info.py             # Model comparison table + feature importance chart
│
├── utils/
│   ├── watchlist.py                # load_watchlist() / save_watchlist() using JSON
│   └── chart_helpers.py            # build_chart() — Plotly chart builder
│
├── data/
│   └── watchlist.json              # Persisted list of saved stock symbols
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── INTERVIEW_NOTES.md              # Likely interview Q&A about this project
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.9 or higher
- Internet connection (for live stock data from Yahoo Finance)

### Step-by-step

```
# 1. Navigate into the project folder
cd "stock-predictor-python"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the models and generate metrics.json
python ml_service/train_model.py

# 6. Launch the Streamlit app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

> **Note:** Step 5 fetches about 2 years of `RELIANCE.NS` data, trains the models,
> and saves them along with `metrics.json` inside `ml_service/`.
> This takes roughly 30–60 seconds on the first run.

---

## 📸 Screenshots

> *[Add a screenshot of the Dashboard page here]*
> *[Add a screenshot of the Model Info page here]*

---

## 📝 Key Design Decisions

### Why predict the return instead of the price?

The first version used absolute prices as features and target, with a model trained on
`RELIANCE.NS`. When used on `IRFC.NS` (a much cheaper stock), the predicted line sat
about Rs 25–28 above the actual line all the time, because the model had learned
Reliance's price level.

The fix: use **scale-free features** (SMA20/Close, SMA50/Close, EMA20/Close, RSI) and
predict the **next-day return**, then convert it to a price with `Close × (1 + return)`.
After this change, the average gap between predicted and actual over 120 days on
`IRFC.NS` dropped from about Rs 25–28 to about Rs 0.27.

### Why time-ordered train/test split (no shuffling)?

Stock data is sequential. Shuffling would let the model see "future" prices during
training — a classic mistake called **data leakage**. We always use the oldest 80%
for training and the newest 20% for testing.

### Why a naive baseline?

Before claiming an ML model is "good", you must prove it beats the simplest possible guess.
Here, the naive baseline is: "tomorrow's price = today's close" (a return of 0).
In this project, neither model beats it, and that result is reported openly.

### Why three metrics?

- **RMSE** — penalises large errors more heavily (important in finance)
- **MAE** — average error in plain rupees (easy to explain to anyone)
- **R²** — shows what % of the variation the model can explain

### Why RSI as a feature?

RSI (Relative Strength Index) is a momentum indicator widely used in technical analysis.
Adding it gives the model information about whether a stock is potentially overbought
or oversold — a domain-knowledge feature that goes beyond pure price history.

---

## ⚠️ Disclaimer

This tool is for **learning and portfolio demonstration only**.
It is not financial advice and should not be used to make real investment decisions.