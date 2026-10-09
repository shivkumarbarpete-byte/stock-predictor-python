"""
pages/3_Model_Info.py
---------------------
Shows how the models were trained and how well they perform.

Sections:
  1. Model comparison table  — Naive vs Linear Regression vs Random Forest
  2. Best model metrics      — shown as st.metric cards
  3. Feature importance chart — Plotly bar chart from Random Forest
  4. Honest disclaimer       — stock prediction is hard; this is for learning

Where does the data come from?
  metrics.json (written by train_model.py). We read it here so there's no
  need to load the model — no slow imports, no memory usage.
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import the path to metrics.json from the central config
from ml_service.config import METRICS_PATH

# --- Page config ---
st.set_page_config(
    page_title="Model Info — Stock Predictor",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Model Performance & Comparison")
st.caption("All metrics are computed on the held-out test set (last 20% of data, time-ordered).")

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# LOAD metrics.json
# ──────────────────────────────────────────────────────────────────────────────

if not METRICS_PATH.exists():
    # Friendly message if the user hasn't trained yet
    st.warning(
        "⚠️ `metrics.json` not found. Models have not been trained yet.\n\n"
        "Please run the following command in your terminal, then refresh this page:\n\n"
        "```\npython ml_service/train_model.py\n```"
    )
    st.stop()   # stop rendering the rest of the page

# Load the JSON file
with open(METRICS_PATH, "r") as f:
    metrics_data = json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Training summary
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("📋 Training Summary")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Trained On",    metrics_data.get("trained_on", "—"))
with s2:
    st.metric("Train Date",    metrics_data.get("train_date", "—"))
with s3:
    st.metric("Training Rows", metrics_data.get("train_rows", "—"))
with s4:
    st.metric("Test Rows",     metrics_data.get("test_rows", "—"))

st.caption(
    f"**Features used:** {', '.join(metrics_data.get('features', []))}"
)

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Model comparison table
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("📊 Model Comparison (Test Set)")
st.markdown(
    "The **Naive Baseline** (no ML at all) sets the bar. "
    "Our models must beat it to prove they've actually learned something."
)

# Build a DataFrame from the metrics list for a clean table display
models_list = metrics_data.get("models", [])
comparison_df = pd.DataFrame([
    {
        "Model":    m["name"],
        "RMSE (₹)": round(m["rmse"], 2),
        "MAE (₹)":  round(m["mae"],  2),
        "R² Score": round(m["r2"],   4),
    }
    for m in models_list
])

# Highlight the best (lowest RMSE) row by finding its index
best_rmse_idx = comparison_df["RMSE (₹)"].idxmin()

# Style the table — bold the best model row
def highlight_best(row):
    """Returns CSS style for each row. Highlights the lowest-RMSE row in green."""
    if row.name == best_rmse_idx:
        return ["background-color: #d4edda; font-weight: bold"] * len(row)
    return [""] * len(row)

styled_table = comparison_df.style.apply(highlight_best, axis=1)
st.dataframe(styled_table, use_container_width=True, hide_index=True)

# ── Best model metrics as st.metric cards ──
best_model = models_list[best_rmse_idx]

st.markdown(f"**Best model: {best_model['name']}** (highlighted above, lowest RMSE)")

b1, b2, b3 = st.columns(3)
with b1:
    st.metric(
        "✅ Best RMSE",
        f"₹{best_model['rmse']:.2f}",
        help="Root Mean Squared Error on the test set — lower is better.",
    )
with b2:
    st.metric(
        "✅ Best MAE",
        f"₹{best_model['mae']:.2f}",
        help="Mean Absolute Error — average prediction error in rupees.",
    )
with b3:
    st.metric(
        "✅ Best R²",
        f"{best_model['r2']:.4f}",
        help="R² = 1.0 is perfect. Tells you how much of price variation the model explains.",
    )

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Feature importance chart (Random Forest)
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("🌲 Random Forest — Feature Importances")
st.markdown(
    """
    Random Forest assigns an **importance score** to each input feature.
    The score shows how much that feature helped improve predictions (all scores add up to 1.0).

    **How to read this chart in an interview:**
    > *"Close price dominates because stock prices are highly autocorrelated — yesterday's
    > price is the strongest predictor of tomorrow's price. The technical indicators
    > contribute smaller but non-zero importance, confirming they add signal."*
    """
)

fi = metrics_data.get("feature_importances", {})

if fi:
    # Sort features from most to least important
    features_sorted     = sorted(fi.keys(), key=lambda k: fi[k], reverse=True)
    importances_sorted  = [fi[f] for f in features_sorted]

    # Plotly horizontal bar chart
    fig_fi = go.Figure(go.Bar(
        x=importances_sorted,
        y=features_sorted,
        orientation="h",                        # horizontal bars
        marker_color=["#22C55E" if f == features_sorted[0] else "#60A5FA"
                      for f in features_sorted],  # top feature in green, rest in blue
        text=[f"{v:.4f}" for v in importances_sorted],
        textposition="outside",
    ))

    fig_fi.update_layout(
        title="Feature Importances (Random Forest)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_white",
        height=350,
        margin=dict(l=80, r=60, t=50, b=40),
        xaxis=dict(range=[0, max(importances_sorted) * 1.2]),  # extra space for labels
    )

    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("Feature importances not found in metrics.json. Re-run train_model.py.")

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Plain-English explanation of what this model can and cannot do
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("⚠️ Honest Disclaimer — Read Before Using")

st.error(
    "**This tool is for LEARNING data science, not for real trading decisions.**\n\n"
    "Stock prices are influenced by thousands of factors: earnings reports, geopolitical events, "
    "market sentiment, institutional buying — none of which this model can see. "
    "A model that looks highly accurate on historical data can still fail on future data."
)

st.markdown(
    """
    #### Why even a high R² doesn't mean the model is reliable for trading

    - **High autocorrelation**: Stock prices tomorrow are very close to today's price.
      Even the naive baseline (tomorrow = today) gets a high R². This inflates our metrics.
    - **No external data**: The model only sees price history. News, fundamentals,
      and macro events are invisible to it.
    - **Regime changes**: A model trained on 2022–2024 data may behave differently in a
      market crisis or bull run it has never seen.
    - **This project's purpose**: To learn and demonstrate proper ML practices —
      time-ordered splits, baseline comparisons, multiple metrics, feature engineering.

    #### What this project DOES demonstrate (interview-worthy)
    - ✅ Avoiding data leakage with a time-ordered train/test split
    - ✅ Always comparing against a naive baseline before claiming "our model is good"
    - ✅ Reporting three metrics (RMSE, MAE, R²) instead of just one
    - ✅ Feature engineering: using domain knowledge (SMA, EMA, RSI) to add signal
    - ✅ Model comparison: Linear Regression vs Random Forest
    """
)
