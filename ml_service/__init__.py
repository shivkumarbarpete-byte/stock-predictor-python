"""
ml_service package
------------------
Reusable modules for data fetching, feature engineering, models,
sentiment analysis, risk-reward calculation, and report generation.
"""

from ml_service.stock_utils import predict_next_close, get_history_with_predictions, fetch_stock_data
from ml_service.news_sentiment import fetch_stock_news, analyze_news_sentiment
from ml_service.risk_reward import calculate_risk_reward, suggest_trade_levels, render_risk_reward_widget
from ml_service.pdf_report import generate_dashboard_pdf, generate_backtest_pdf
