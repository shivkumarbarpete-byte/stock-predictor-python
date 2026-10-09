# Stock Market Analysis & Prediction Tool — Pure Python Conversion Plan

---

## Part 1 — What the Existing ML Code Does (Simple English)

Think of the code as a **4-step assembly line**:

### Step 1 · `fetch_data.py` → Get the raw data
- Connects to **Yahoo Finance** using the `yfinance` library.
- Downloads daily stock price history (Open, High, Low, Close, Volume) for the last 2 years for any NSE stock (e.g. `RELIANCE.NS`).
- Saves a clean table (DataFrame) — like a spreadsheet — with one row per trading day.
- Has a **10-minute in-memory cache** in `stock_utils.py` so it doesn't spam Yahoo Finance on every request.

### Step 2 · `compute_indicators.py` → Calculate technical signals
Three columns are added to the table:
| Indicator | What it is | Window |
|---|---|---|
| **SMA20** | Average closing price of last 20 days | 20 days |
| **SMA50** | Average closing price of last 50 days | 50 days |
| **EMA20** | Like SMA but recent days count more | 20 days (exponential) |

A bonus column `Trend_Signal` (1 = bullish, 0 = bearish) is computed by checking if SMA20 > SMA50.
The first ~49 rows don't have enough history for SMA50, so they are dropped before training.

### Step 3 · `train_model.py` → Train and save the ML model
- **Target** (what we want to predict): tomorrow's Close price — created by shifting today's Close up by 1 row.
- **Features** (inputs to the model): `[Close, SMA20, SMA50, EMA20]` — 4 numbers per day.
- **Split**: 80% of data is used for training, last 20% for testing — in time order (oldest to newest), **not** randomly shuffled. This is correct and avoids "data leakage" (accidentally letting the model peek at future prices).
- **Model**: `LinearRegression` from scikit-learn — finds the best-fit straight line through the feature space.
- **Metrics calculated**: RMSE (average error in rupees) and R² (how well the model explains price movement).
- The trained model is saved to `linear_regression_model.pkl` using `joblib`.

### Step 4 · `stock_utils.py` + `main.py` → Serve via FastAPI
- `stock_utils.py` bundles all reusable logic: fetch → add indicators → load model → predict.
- `main.py` wraps this in a **REST API** with three endpoints:
  - `/predict/{symbol}` → next-day predicted price + metrics
  - `/history/{symbol}?days=120` → last N days of actual vs predicted prices (for the chart)
  - `/quote/{symbol}` → latest price snapshot

**The FastAPI layer is what we're replacing with Streamlit.** All the ML logic in `stock_utils.py` stays.

---

## Part 2 — Pure Python Architecture

### Why drop Node / Express / React / MongoDB?

| Old piece | Why it existed | Pure Python replacement |
|---|---|---|
| Express + JWT | Auth API server | **Not needed** (see below) |
| MongoDB | Store watchlists per user | **Local JSON file** |
| React | UI / charts | **Streamlit** |
| Axios calls | Talk to FastAPI | **Direct Python function calls** |

> [!IMPORTANT]
> **Do you really need login/auth?**
> 
> For a data science portfolio project: **NO, drop it completely.** Here's why:
> - You're running this locally on your own machine, not deploying to thousands of users.
> - A login system adds ~200 lines of code (JWT, hashing, sessions) that are not ML/DS-related.
> - Interviewers care about your ML pipeline, not auth middleware.
> - **Recommendation**: Skip auth entirely. One watchlist, saved locally. If you ever do want it, add it later as a separate task.

### Watchlist Storage: JSON file vs SQLite

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **JSON file** (`watchlist.json`) | Zero setup, human-readable, 5 lines of code | Not suitable for large data | ✅ **Use this** |
| SQLite | Proper database, good for learning | Needs schema, more code | Overkill for a watchlist |

A watchlist is just a list of ticker symbols: `["RELIANCE.NS", "TCS.NS", "INFY.NS"]`. A JSON file is perfect.

---

## Part 3 — Clean Folder Structure

```
stock-predictor-python/
│
├── app.py                        ← Streamlit entry point (python -m streamlit run app.py)
│
├── ml_service/                   ← Renamed from ml-service (hyphens cause Python import issues)
│   ├── stock_utils.py            ← REUSED AS-IS: fetch, indicators, predict, history
│   ├── train_model.py            ← REUSED AS-IS: training script (run once manually)
│   ├── compute_indicators.py     ← REUSED AS-IS: SMA/EMA logic
│   ├── fetch_data.py             ← REUSED AS-IS: yfinance wrapper
│   └── linear_regression_model.pkl  ← REUSED AS-IS: saved trained model
│
├── pages/                        ← Streamlit multi-page support (auto-discovered)
│   ├── 1_Dashboard.py            ← Main chart page: search + Plotly chart + metrics
│   ├── 2_Watchlist.py            ← Add/remove/view saved stocks
│   └── 3_Model_Info.py           ← Show RMSE, R², feature importance (good for interviews)
│
├── utils/
│   ├── watchlist.py              ← Read/write watchlist.json (5 lines of code)
│   └── chart_helpers.py          ← Build the Plotly figure (keeps app.py clean)
│
├── data/
│   ├── watchlist.json            ← Persisted list: ["RELIANCE.NS", "TCS.NS"]
│   └── (CSV files from ml-service can stay here or in ml_service/)
│
├── requirements.txt              ← Updated: swap fastapi+uvicorn for streamlit+plotly
└── README.md                     ← How to run the project
```

### One-line explanation of each file

| File | What it does |
|---|---|
| `app.py` | Launches the Streamlit app; shows the sidebar navigation |
| `ml_service/stock_utils.py` | **Core brain**: fetches data, computes indicators, loads model, predicts — reused from old code |
| `ml_service/train_model.py` | Run this once to train/retrain the model and save the `.pkl` file |
| `ml_service/compute_indicators.py` | Adds SMA20, SMA50, EMA20 columns to a DataFrame |
| `ml_service/fetch_data.py` | Downloads stock history from Yahoo Finance via yfinance |
| `pages/1_Dashboard.py` | The main UI: search box, Plotly chart (actual + predicted + MAs), next-day prediction card |
| `pages/2_Watchlist.py` | UI to add/remove stocks from watchlist; lists saved stocks with quick stats |
| `pages/3_Model_Info.py` | Shows model metrics (RMSE, MAE, R²), feature importance bar chart — great for interviews |
| `utils/watchlist.py` | Tiny helper: `load_watchlist()` and `save_watchlist()` using `watchlist.json` |
| `utils/chart_helpers.py` | Builds the multi-line Plotly chart — keeps the page files short and readable |
| `data/watchlist.json` | Simple JSON list of saved tickers, e.g. `["RELIANCE.NS", "TCS.NS"]` |
| `requirements.txt` | Lists all Python dependencies (streamlit, plotly, yfinance, scikit-learn, pandas, joblib) |

---

## Part 4 — ML Improvements for Data Science Interviews

These are ranked from most impactful (for interviews) to "nice to have":

### 🥇 #1 — Add RSI (Relative Strength Index) as a feature
**Beginner-friendly? ✅ Yes**

RSI is a momentum indicator (0–100 scale). Below 30 = stock likely oversold (good buy signal), above 70 = overbought. Adding it as a feature might improve predictions AND shows interviewers you know real trading indicators.

```python
# Add to compute_indicators.py
def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df
```

Then add `"RSI"` to the `FEATURES` list and retrain.

---

### 🥈 #2 — Compare Linear Regression vs Random Forest (model comparison)
**Beginner-friendly? ✅ Yes, with guidance**

Training two models and comparing their RMSE/R² side by side is a classic interview talking point. Random Forest often wins on stock data because it captures non-linear relationships.

```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

Show a comparison table in the `Model_Info` page:
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 45.20 | 32.10 | 0.9821 |
| Random Forest | 38.60 | 27.40 | 0.9889 |

---

### 🥉 #3 — Add MAE metric and a proper metrics dashboard
**Beginner-friendly? ✅ Very easy — already partially in `stock_utils.py`**

MAE (Mean Absolute Error) is already computed in `stock_utils.py`. You just need to display it nicely. Good interviewers ask: *"Why RMSE and not just MAE?"* — because RMSE penalizes large errors more heavily (important in finance where one bad prediction can be costly).

The `stock_utils.py` already computes all three metrics (`mae`, `rmse`, `r2`) — just surface them on the Streamlit page with `st.metric()`.

---

### 🎯 #4 — Add a "Prediction Confidence" indicator (Advanced but impressive)
**Beginner-friendly? ⚠️ Moderate**

Instead of just one predicted number, show a prediction range (e.g. "₹2,450 ± ₹45"). You can estimate this from the historical RMSE — it's not true confidence intervals but it looks great on a dashboard and shows statistical thinking.

```python
predicted = 2450.00
margin = rmse  # e.g. 45.20
lower = predicted - margin
upper = predicted + margin
# Display: "₹2,404.80 — ₹2,495.20"
```

---

## Part 5 — Step-by-Step Build Order

Follow these one at a time. Don't skip ahead — each step tests the previous one.

```
Step 1 ─── Rename & Reorganise
           Rename ml-service/ → ml_service/ (fix the hyphen)
           Create the folders: pages/, utils/, data/
           Move the CSV files to data/

Step 2 ─── Fix imports & verify ML code still works
           Update imports in stock_utils.py to use the new folder name
           Run: python ml_service/stock_utils.py
           Expected: Should print prediction output for RELIANCE.NS

Step 3 ─── Install Streamlit + Plotly, update requirements.txt
           pip install streamlit plotly
           Remove fastapi and uvicorn from requirements.txt

Step 4 ─── Build utils/watchlist.py (5 lines)
           load_watchlist() and save_watchlist() functions
           Test it in a Python shell: add a ticker, reload, check the JSON file

Step 5 ─── Build utils/chart_helpers.py
           A single function: build_chart(df) → returns a Plotly figure
           Plots: Actual (solid), Predicted (dashed), SMA20 (dotted), SMA50 (dotted)

Step 6 ─── Build pages/1_Dashboard.py (the main chart page)
           Search box → fetch data → build chart → show next-day prediction
           Test: streamlit run app.py

Step 7 ─── Build pages/2_Watchlist.py
           List saved stocks, button to add current stock, button to remove

Step 8 ─── Add RSI to compute_indicators.py and retrain the model
           python ml_service/train_model.py  (saves a new .pkl)
           Verify RMSE improved

Step 9 ─── Build pages/3_Model_Info.py
           Show RMSE / MAE / R² as st.metric() boxes
           Add a bar chart of feature importances (Random Forest gives these for free)
           Add model comparison table (Linear Regression vs Random Forest)

Step 10 ── Polish & README
           Add a sidebar logo or header
           Write README.md with setup instructions
           Take a screenshot for your portfolio / resume
```

---

> [!TIP]
> **Interview talking points this project gives you:**
> - "I used time-ordered train/test split to avoid data leakage" ← shows you understand the difference from random split
> - "I compared Linear Regression and Random Forest and measured RMSE, MAE, R²" ← shows model evaluation skills
> - "I added RSI as a feature because momentum is a known signal in financial literature" ← shows domain knowledge
> - "I replaced a full MERN stack with a pure Python pipeline to reduce complexity" ← shows architectural judgment

