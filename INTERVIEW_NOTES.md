# 🎯 Interview Notes — Stock Market Prediction Project

These are **12 likely interview questions** about this project, with honest, beginner-level answers.
The goal is not to memorise these — it's to understand them well enough to answer naturally.

> **Important:** Metric values (RMSE, R², etc.) are marked as `[see metrics.json]`.
> Run `python ml_service/train_model.py` first, then replace the placeholders with your actual numbers.

---

## 1. What is data leakage, and how did you avoid it in this project?

**Data leakage** means accidentally giving the model information from the future
during training — so it learns "answers it shouldn't know" and looks great on
training data but fails in real life.

**How I avoided it:**
- I used a **time-ordered split**: the first 80% of data (oldest dates) is used for
  training, and the last 20% (newest dates) is used for testing.
- I **never shuffled** the data. Shuffling would mix old and new dates, so the model
  could train on data from, say, March 2024 and test on January 2024 —
  effectively "seeing the future."

---

## 2. Why do you split by time instead of using a random train/test split?

Stock prices are **sequential** — tomorrow's price depends on today's.
A random split breaks this time order, so the model trains on data from the future
and tests on the past. It would look artificially accurate on the test set but
fail completely in real use.

The rule for any **time-series problem**: always split by time, never shuffle.

---

## 3. What is RMSE and how is it different from MAE?

- **MAE (Mean Absolute Error)**: average of `|actual - predicted|`.
  Easy to explain: "on average, the model's predictions are off by ₹X."

- **RMSE (Root Mean Squared Error)**: square the errors before averaging,
  then take the square root. Because of the squaring, **large errors are penalised
  more heavily** than small ones.

**When does this matter?**
In finance, one very wrong prediction can be costly. RMSE reflects that risk better.
If your RMSE is much larger than your MAE, it means the model occasionally makes
very large errors — a red flag worth investigating.

For this project: MAE = `[see metrics.json]`, RMSE = `[see metrics.json]`.

---

## 4. Why did you add a naive baseline? What is it?

The **naive baseline** is the simplest possible "model": predict that tomorrow's
closing price equals today's closing price. No ML involved at all.

**Why it matters:**
Before claiming your ML model is good, you must prove it is better than the dumbest
possible guess. If Random Forest barely beats the naive baseline, it hasn't learned
anything useful — it's just tracking price momentum, not finding real patterns.

Interviewers respect this because many beginners skip the baseline comparison.

In this project: Naive RMSE = `[see metrics.json]`, LR RMSE = `[see metrics.json]`,
RF RMSE = `[see metrics.json]`.

---

## 5. What does R² (R-squared) score mean? Why can it be misleading here?

**R² score** measures how much of the variation in the target variable
(tomorrow's close price) the model explains.

- R² = 1.0 → perfect predictions
- R² = 0.0 → model is no better than always predicting the average price
- R² can be negative → model is worse than predicting the average

**Why it can be misleading for stock data:**
Stock prices tomorrow are always very close to today's price — the naive baseline
already has a very high R² because of this **autocorrelation**. A high R² does not
mean the model is actually useful for trading.

---

## 6. What is RSI and why did you add it as a feature?

**RSI (Relative Strength Index)** is a 14-day momentum indicator (range 0–100).

- RSI > 70 → the stock has risen fast recently, possibly **overbought** (may fall)
- RSI < 30 → the stock has fallen fast recently, possibly **oversold** (may rise)
- RSI = 50 → neutral momentum

**Why I added it:**
SMA and EMA only capture trend direction. RSI adds information about
**speed of price movement** — a different kind of signal. This is domain knowledge
from technical analysis, which is a good thing to mention in an interview because
it shows you understand the problem, not just the code.

---

## 7. What is the difference between SMA and EMA?

| | SMA (Simple Moving Average) | EMA (Exponential Moving Average) |
|---|---|---|
| Calculation | Plain average of last N prices | Weighted average — recent prices count more |
| Lag | Higher (responds slowly to price changes) | Lower (responds faster) |
| Use | Long-term trend (SMA50) | Short-term momentum (EMA20) |

**In this project:** SMA20, SMA50 for trend, EMA20 for short-term movement.
The gap between SMA20 and SMA50 is a classic **crossover signal** used in trading.

---

## 8. Why does Random Forest often outperform Linear Regression on this data?

**Linear Regression** assumes a straight-line (linear) relationship between
features and the target. Stock price relationships are not always linear
(e.g. RSI's effect might be different when prices are trending up vs down).

**Random Forest** builds 100 decision trees on random subsets of the data and
averages their predictions. It can capture **non-linear patterns** and **feature
interactions** that Linear Regression misses.

The cost: Random Forest is harder to explain mathematically (it's a "black box"
compared to Linear Regression's simple formula).

---

## 9. What is overfitting? How would you detect it in this project?

**Overfitting** means the model memorises the training data so well that it
fails on new data it hasn't seen.

**Signs of overfitting:**
- Training RMSE is much lower than test RMSE (the model "cheated" on training data)
- Very high R² on training, much lower R² on the test set

**How this project guards against it:**
- We never look at test-set metrics during training
- We do a proper time-ordered split (no data leakage)
- Random Forest's `n_estimators=100` and `random_state=42` give stable results

**To improve further:** use cross-validation (time-series cross-validation, not k-fold)
and add regularisation (e.g. Ridge Regression instead of plain Linear Regression).

---

## 10. What are the limitations of this stock prediction model?

1. **Only sees price history** — no earnings reports, no news, no fundamentals
2. **Trained on one stock (RELIANCE.NS)** — predictions for other stocks use the
   same model weights, which may not generalise well
3. **Assumes past patterns repeat** — market regime changes (e.g. a crash, a bull run)
   can invalidate a model trained in "normal" conditions
4. **Short prediction horizon** — predicting one day ahead. Beyond 1–2 days,
   errors compound rapidly
5. **High autocorrelation inflates metrics** — tomorrow's price ≈ today's price
   for nearly any stock, so even a naive baseline looks good numerically

---

## 11. How would you improve this model next?

**Easy improvements (beginner-friendly):**
- Add more features: MACD (Moving Average Convergence Divergence), Bollinger Bands,
  trading volume change
- Train a separate model for each stock instead of one model for all stocks
- Add `Walk-Forward Validation` (a time-series-specific cross-validation method)

**Intermediate improvements:**
- Use `Ridge Regression` or `Lasso` instead of plain Linear Regression
  (adds regularisation to reduce overfitting)
- Try `XGBoost` or `LightGBM` — gradient-boosted trees that often beat Random Forest

**Advanced improvements:**
- Add sentiment analysis from financial news headlines as a feature
- Try LSTM (Long Short-Term Memory) neural networks — designed for sequence prediction

---

## 12. Why not use k-fold cross-validation? Why time-series cross-validation instead?

**Standard k-fold cross-validation** randomly splits the data into K folds.
For time-series data this creates leakage — some folds will train on "future" data
and test on "past" data.

**Time-series cross-validation (Walk-Forward Validation)** always trains on the past
and tests on the immediate future:
- Fold 1: train on months 1–6, test on month 7
- Fold 2: train on months 1–7, test on month 8
- …and so on

This gives a more realistic picture of how the model would perform in real deployment.
For this project, a single time-ordered 80/20 split is used for simplicity —
worth mentioning walk-forward as a natural next improvement.

---

## 💡 One-sentence answers for quick rounds

| Question | One-line answer |
|---|---|
| Why time-ordered split? | "Shuffling creates data leakage — the model sees future prices during training." |
| Why baseline? | "You must prove ML is better than the dumbest possible guess before claiming it works." |
| RMSE vs MAE? | "RMSE penalises large errors more; MAE is the plain average error in rupees." |
| Why RSI? | "It adds momentum information — whether the stock is overbought or oversold." |
| Why RF over LR? | "RF captures non-linear patterns; LR assumes a straight-line relationship." |
| Biggest limitation? | "The model only sees price history — news, fundamentals, and events are invisible to it." |
