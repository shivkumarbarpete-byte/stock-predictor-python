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
from ml_service.validation import run_walk_forward_validation

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
# SECTION 2 — Model comparison table & Directional Accuracy
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
        help="R² on price. High price R² is expected because tomorrow's price is close to today's.",
    )

# ── Directional accuracy for LR and RF vs 50% coin-flip line ──
st.markdown("---")
st.subheader("🎯 Directional Accuracy vs 50% Coin-Flip Line")
st.caption(
    "Measures the % of test days the model correctly called the market direction (Up vs Down). "
    "A pure coin flip gives 50%. A useful model must consistently beat 50%."
)

extra_m = metrics_data.get("extra_metrics", {})
lr_dir = extra_m.get("lr", {}).get("dir_acc")
rf_dir = extra_m.get("rf", {}).get("dir_acc")
lr_dir_pct = (lr_dir * 100) if lr_dir is not None else 0.0
rf_dir_pct = (rf_dir * 100) if rf_dir is not None else 0.0

col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    st.metric(
        "🪙 Coin-Flip Baseline",
        "50.0%",
        help="Pure random guessing baseline.",
    )
with col_d2:
    delta_lr = lr_dir_pct - 50.0
    st.metric(
        "Linear Regression",
        f"{lr_dir_pct:.1f}%",
        delta=f"{delta_lr:+.1f}pp vs coin flip",
        delta_color="normal" if delta_lr >= 0 else "inverse",
        help="Test-set directional accuracy for Linear Regression.",
    )
with col_d3:
    delta_rf = rf_dir_pct - 50.0
    st.metric(
        "Random Forest",
        f"{rf_dir_pct:.1f}%",
        delta=f"{delta_rf:+.1f}pp vs coin flip",
        delta_color="normal" if delta_rf >= 0 else "inverse",
        help="Test-set directional accuracy for Random Forest.",
    )

# Visual Plotly chart with 50% coin-flip reference line
fig_dir = go.Figure()
fig_dir.add_trace(go.Bar(
    x=["Linear Regression", "Random Forest"],
    y=[lr_dir_pct, rf_dir_pct],
    marker_color=["#3B82F6", "#10B981"],
    text=[f"{lr_dir_pct:.1f}%", f"{rf_dir_pct:.1f}%"],
    textposition="outside",
    width=0.35,
    name="Model Accuracy",
))
fig_dir.add_hline(
    y=50,
    line_dash="dash",
    line_color="#EF4444",
    line_width=2,
    annotation_text="50% Coin-Flip Line",
    annotation_position="top right",
    annotation_font_color="#EF4444",
)
fig_dir.update_layout(
    title="Directional Accuracy on Held-Out Test Set",
    yaxis=dict(title="Accuracy (%)", range=[0, 100]),
    xaxis=dict(title="Model"),
    template="plotly_white",
    height=320,
    margin=dict(l=60, r=40, t=50, b=40),
)
st.plotly_chart(fig_dir, use_container_width=True)

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Walk-Forward Validation (Expanding Window, 5 Folds)
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("🔄 Walk-Forward Validation (Expanding Window, 5 Folds)")
st.markdown(
    "A single train/test split can be lucky or unlucky depending on market conditions. "
    "**Walk-forward validation** (time-ordered, expanding window, 5 folds, no shuffling) tests "
    "the models across 5 distinct sequential time windows."
)

with st.spinner("Loading walk-forward validation results..."):
    wf_raw = run_walk_forward_validation(n_splits=5)

# Build a formatted copy with summary row
display_wf = wf_raw.copy()

# Add an Average summary row
avg_row = {
    "Fold": "**Average (5 Folds)**",
    "Test Period": "All test windows",
    "Train / Test Size": "—",
    "Naive Return MAE": display_wf["Naive Return MAE"].mean(),
    "LR Return MAE": display_wf["LR Return MAE"].mean(),
    "RF Return MAE": display_wf["RF Return MAE"].mean(),
    "LR Dir. Accuracy": display_wf["LR Dir. Accuracy"].mean(),
    "RF Dir. Accuracy": display_wf["RF Dir. Accuracy"].mean(),
}
display_wf = pd.concat([display_wf, pd.DataFrame([avg_row])], ignore_index=True)

# Format floats for clean presentation
formatted_wf = pd.DataFrame({
    "Fold": display_wf["Fold"],
    "Test Window": display_wf["Test Period"],
    "Train / Test Size": display_wf["Train / Test Size"],
    "Naive Return MAE": display_wf["Naive Return MAE"].apply(lambda v: f"{v:.4f} ({v*100:.2f}%)"),
    "LR Return MAE": display_wf["LR Return MAE"].apply(lambda v: f"{v:.4f} ({v*100:.2f}%)"),
    "RF Return MAE": display_wf["RF Return MAE"].apply(lambda v: f"{v:.4f} ({v*100:.2f}%)"),
    "LR Dir. Acc": display_wf["LR Dir. Accuracy"].apply(lambda v: f"{v:.1f}%"),
    "RF Dir. Acc": display_wf["RF Dir. Accuracy"].apply(lambda v: f"{v:.1f}%"),
})

st.dataframe(formatted_wf, use_container_width=True, hide_index=True)

# Plain-English "How to read these results" box
st.info(
    "💡 **How to read these results (Interview Takeaway):**\n\n"
    "1. **The models do not beat the naive baseline:** Across folds, the Naive baseline "
    "(`tomorrow's return = 0`, or `tomorrow = today`) consistently achieves a Return MAE equal to or lower "
    "than both Linear Regression and Random Forest.\n"
    "2. **Directional accuracy hovers near ~50% (coin flip):** The models cannot reliably call whether "
    "tomorrow will be an Up or Down day better than random chance.\n"
    "3. **Why this is completely expected:** In liquid financial markets, day-to-day returns approximate a random walk "
    "(weak-form Market Efficiency). Lagged technical indicators alone do not provide persistent alpha over 1-day horizons. "
    "Recognizing and demonstrating this honest evaluation is precisely what quantitative interviewers look for."
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
