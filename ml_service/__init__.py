"""
ml_service package
------------------
Reusable modules for data fetching, feature engineering, models,
sentiment analysis, risk-reward calculation, validation, and report generation.
"""

from ml_service.stock_utils import (
    predict_next_close,
    get_history_with_predictions,
    fetch_stock_data,
    load_model,
    get_available_models,
)
from ml_service.news_sentiment import fetch_stock_news, analyze_news_sentiment
from ml_service.risk_reward import (
    calculate_risk_reward,
    suggest_trade_levels,
    render_risk_reward_widget,
    compute_trading_metrics,
    compute_drawdown_series,
    compute_max_drawdown,
)
from ml_service.validation import (
    validate_ticker_symbol,
    safe_fetch_stock_summary,
    run_walk_forward_validation,
)
from ml_service.pdf_report import (
    generate_dashboard_pdf,
    generate_backtest_pdf,
    generate_watchlist_pdf,
)
from ml_service.backtest import run_backtest
