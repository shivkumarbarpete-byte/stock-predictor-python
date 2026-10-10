"""
ml_service/config.py
--------------------
SINGLE SOURCE OF TRUTH for features and file paths.

The rule: EVERY file that trains or predicts (train_model.py, stock_utils.py,
pages/1_Dashboard.py) imports FEATURES from HERE.

Why this matters:
  If you add a new feature (like RSI) and forget to update one of those files,
  the model will be trained on 5 features but predict with 4 (or vice versa)
  and Python will throw a confusing shape-mismatch error.
  Having one config file makes it impossible to go out of sync.

To add a new feature in the future:
  1. Add it here in FEATURES.
  2. Add the calculation in stock_utils.py -> add_indicators().
  3. Retrain: python ml_service/train_model.py
  That's it — train_model.py and stock_utils.py both import from here.
"""

from pathlib import Path

# ---- Directory where this file lives (i.e. ml_service/) ----
# Path(__file__).parent always points to the folder containing THIS file,
# no matter where you run the app from.
ML_DIR = Path(__file__).parent


# ---- Feature list ----
# All features are SCALE-FREE (ratios / RSI), so the same model works for
# a Rs 80 stock and a Rs 1300 stock. Raw "Close" is NOT a feature anymore.
# Order matters: must be the same during training AND during prediction.
FEATURES = ["SMA20_ratio", "SMA50_ratio", "EMA20_ratio", "RSI"]

# ---- Model file paths ----
# v3 = scale-free models (predict next-day RETURN). Older v2/original .pkl
# files are kept untouched.
MODEL_LR_PATH = ML_DIR / "linear_regression_model_v3.pkl"
MODEL_RF_PATH = ML_DIR / "random_forest_model_v3.pkl"

# ---- Metrics file path ----
# train_model.py writes here; pages/3_Model_Info.py reads from here.
METRICS_PATH = ML_DIR / "metrics.json"
