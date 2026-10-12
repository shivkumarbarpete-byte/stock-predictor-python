"""
ml_service/pdf_report.py
------------------------
Generates downloadable, cleanly formatted PDF reports for the Stock Predictor:
  1. Dashboard Analysis Report (Executive summary, predictions, indicators,
     model quality, news sentiment, risk/reward setup).
  2. Backtest Strategy Report (Performance table, equity summary, trade log,
     methodology and leakage disclaimers).

Built using fpdf2 with bulletproof text sanitization for currency and symbols.
"""

from datetime import datetime
from typing import Dict, List, Optional
from fpdf import FPDF


def sanitize_text(text: str) -> str:
    """
    Sanitizes string for standard PDF fonts (Latin-1).
    Converts Indian Rupee symbol to Rs., normalizes dashes, quotes, and strips emojis.
    """
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "₹": "Rs. ",
        "•": "-",
        "–": "-",
        "—": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "🟢": "[Bullish] ",
        "🔴": "[Bearish] ",
        "🟡": "[Neutral] ",
        "⚪": "[Neutral] ",
        "±": "+/-",
        "≥": ">=",
        "≤": "<=",
        "·": "-",
        "…": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Encode to latin-1 replacing unknown characters with '?' to ensure no crash
    return text.encode("latin-1", "replace").decode("latin-1")


class StockPDFReport(FPDF):
    """Custom styled FPDF report with header, footer, and utility card methods."""

    def __init__(self, title_text: str = "Stock Predictor Report", subtitle_text: str = ""):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.title_text = title_text
        self.subtitle_text = subtitle_text

    def header(self):
        # Top banner background
        self.set_fill_color(30, 41, 59)  # Slate-800
        self.rect(0, 0, 210, 22, "F")

        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.set_xy(10, 5)
        self.cell(120, 6, sanitize_text(self.title_text), align="L")

        self.set_font("Helvetica", "", 9)
        self.set_text_color(203, 213, 225)
        self.set_xy(10, 11)
        self.cell(120, 5, sanitize_text(self.subtitle_text), align="L")

        # Timestamp on top right
        self.set_font("Helvetica", "", 8)
        self.set_xy(135, 7)
        now_str = datetime.now().strftime("%d %b %Y, %H:%M")
        self.cell(65, 5, sanitize_text(f"Generated: {now_str}"), align="R")

        self.set_text_color(0, 0, 0)
        self.set_y(26)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        disclaimer = "Stock Predictor Python | For educational and research purposes only. Not financial advice."
        self.cell(140, 6, sanitize_text(disclaimer), align="L")
        self.cell(50, 6, sanitize_text(f"Page {self.page_no()}"), align="R")

    def draw_section_heading(self, title: str):
        self.ln(2)
        self.set_fill_color(241, 245, 249)  # Slate-100
        self.set_text_color(15, 23, 42)
        self.set_font("Helvetica", "B", 10)
        self.cell(190, 7, sanitize_text(f"  {title}"), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def draw_key_value(self, label: str, value: str, width: float = 95, height: float = 6):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(71, 85, 105)
        self.cell(40, height, sanitize_text(label), align="L")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(15, 23, 42)
        self.cell(width - 40, height, sanitize_text(value), align="L")


def generate_dashboard_pdf(
    symbol: str,
    pred: Dict,
    chart_data: Optional[List[Dict]] = None,
    sentiment_data: Optional[Dict] = None,
    risk_reward_data: Optional[Dict] = None,
) -> bytes:
    """
    Generates a PDF document compiling the stock prediction,
    technical indicators, model metrics, news sentiment, and trade setup.
    """
    model_name_label = "Linear Regression" if pred.get("model_used") == "lr" else "Random Forest"
    pdf = StockPDFReport(
        title_text=f"AI STOCK ANALYSIS REPORT: {symbol}",
        subtitle_text=f"Model: {model_name_label}  |  Last Close: Rs. {pred.get('last_close', 0.0):,.2f}",
    )
    pdf.add_page()

    # 1. Executive Summary & Prediction
    pdf.draw_section_heading("1. EXECUTIVE SUMMARY & TOMORROW'S PREDICTION")

    last_close = pred.get("last_close", 0.0)
    pred_close = pred.get("predicted_close", 0.0)
    change = pred.get("change", 0.0)
    change_pct = pred.get("change_percent", 0.0)
    direction = pred.get("direction", "NEUTRAL")
    rmse = pred.get("metrics", {}).get("rmse", 0.0)
    lower_bound = round(pred_close - rmse, 2)
    upper_bound = round(pred_close + rmse, 2)

    pdf.draw_key_value("Last Trading Date:", pred.get("last_date", "N/A"), width=95)
    pdf.draw_key_value("Direction Signal:", f"{direction} ({change_pct:+.2f}%)", width=95)
    pdf.ln(6)

    pdf.draw_key_value("Current Close Price:", f"Rs. {last_close:,.2f}", width=95)
    pdf.draw_key_value("Predicted Close:", f"Rs. {pred_close:,.2f} ({change:+.2f})", width=95)
    pdf.ln(6)

    pdf.draw_key_value("Prediction Range:", f"Rs. {lower_bound:,.2f} to Rs. {upper_bound:,.2f} (+/- RMSE Rs. {rmse:,.2f})", width=190)
    pdf.ln(8)

    # 2. Technical Indicators
    pdf.draw_section_heading("2. LATEST TECHNICAL INDICATORS")
    ind = pred.get("indicators", {})

    pdf.draw_key_value("SMA 20 (20-day):", f"Rs. {ind.get('SMA20', 0.0):,.2f}", width=95)
    pdf.draw_key_value("SMA 50 (50-day):", f"Rs. {ind.get('SMA50', 0.0):,.2f}", width=95)
    pdf.ln(6)

    rsi_val = ind.get("RSI", 50.0)
    rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    pdf.draw_key_value("EMA 20:", f"Rs. {ind.get('EMA20', 0.0):,.2f}", width=95)
    pdf.draw_key_value("RSI 14:", f"{rsi_val:.1f} ({rsi_status})", width=95)
    pdf.ln(8)

    # 3. Model Quality & Backtest Validation
    pdf.draw_section_heading("3. MODEL VALIDATION ON THIS STOCK (RECENT 20% TEST SET)")
    m = pred.get("metrics", {})

    pdf.draw_key_value("MAE (Model):", f"Rs. {m.get('mae', 0.0):,.2f}", width=95)
    pdf.draw_key_value("MAE (Naive Baseline):", f"Rs. {m.get('naive_mae', 0.0):,.2f}", width=95)
    pdf.ln(6)

    pdf.draw_key_value("RMSE (Model):", f"Rs. {m.get('rmse', 0.0):,.2f}", width=95)
    pdf.draw_key_value("RMSE (Naive Baseline):", f"Rs. {m.get('naive_rmse', 0.0):,.2f}", width=95)
    pdf.ln(6)

    pdf.draw_key_value("R2 on Price:", f"{m.get('r2', 0.0):.4f}", width=95)
    pdf.draw_key_value("R2 on Return (Honest):", f"{m.get('r2_return', 0.0):.4f}", width=95)
    pdf.ln(6)

    dir_acc = m.get("directional_accuracy", 50.0)
    pdf.draw_key_value("Directional Accuracy:", f"{dir_acc:.1f}% (vs 50% random coin-flip)", width=190)
    pdf.ln(8)

    # 4. News Sentiment & Market Tone
    if sentiment_data:
        pdf.draw_section_heading("4. RECENT NEWS SENTIMENT & AI SUMMARY")
        counts = sentiment_data.get("counts", {})
        pdf.draw_key_value(
            "Overall Sentiment:",
            f"{sentiment_data.get('sentiment_label', 'Neutral')} (Score: {sentiment_data.get('overall_score', 0.0):+.2f})",
            width=95
        )
        pdf.draw_key_value(
            "Headlines Breakdown:",
            f"{counts.get('positive', 0)} Positive  |  {counts.get('neutral', 0)} Neutral  |  {counts.get('negative', 0)} Negative",
            width=95
        )
        pdf.ln(6)

        ai_summary = sentiment_data.get("ai_summary", "")
        if ai_summary:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(51, 65, 85)
            # Remove markdown bold stars for clean plain PDF text
            clean_summary = ai_summary.replace("**", "")
            pdf.multi_cell(190, 5, sanitize_text(clean_summary))
            pdf.ln(2)

        # List top 3 headlines
        articles = sentiment_data.get("articles", [])[:3]
        if articles:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(190, 5, sanitize_text("Top Recent Headlines Analyzed:"), new_x="LMARGIN", new_y="NEXT")
            for art in articles:
                pdf.set_font("Helvetica", "", 8)
                score_str = f"[{art.get('label', 'Neutral')}: {art.get('score', 0.0):+.2f}]"
                title_line = f"- {score_str} {art.get('title', '')} ({art.get('publisher', 'News')})"
                pdf.multi_cell(190, 4, sanitize_text(title_line))
            pdf.ln(4)

    # 5. Risk / Reward & Trade Setup
    if risk_reward_data:
        pdf.draw_section_heading("5. RISK / REWARD & POSITION SIZING SETUP")
        pdf.draw_key_value("Entry Price:", f"Rs. {risk_reward_data.get('entry_price', 0.0):,.2f}", width=95)
        pdf.draw_key_value("Stop-Loss Price:", f"Rs. {risk_reward_data.get('stop_loss', 0.0):,.2f} (-{risk_reward_data.get('risk_pct', 0.0):.2f}%)", width=95)
        pdf.ln(6)

        pdf.draw_key_value("Target Price:", f"Rs. {risk_reward_data.get('target', 0.0):,.2f} (+{risk_reward_data.get('reward_pct', 0.0):.2f}%)", width=95)
        pdf.draw_key_value("Risk : Reward Ratio:", f"{risk_reward_data.get('rr_ratio_formatted', 'N/A')} ({risk_reward_data.get('quality_label', '')})", width=95)
        pdf.ln(6)

        pos = risk_reward_data.get("position_sizing", {})
        if pos:
            pos_line = (
                f"Sizing: {pos.get('suggested_shares', 0)} shares "
                f"(Capital: Rs. {pos.get('total_investment', 0.0):,.2f} | "
                f"Risk: Rs. {pos.get('total_risk_rupees', 0.0):,.2f} | "
                f"Target Profit: Rs. {pos.get('total_potential_profit', 0.0):,.2f})"
            )
            pdf.draw_key_value("Position Allocation:", pos_line, width=190)
            pdf.ln(6)

    # Disclaimers note
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(
        190, 4,
        sanitize_text(
            "Methodology Disclaimer: Machine learning models predict next-day expected returns based on historical price "
            "series and technical indicators. Financial markets are subject to macroeconomic volatility and regime shifts. "
            "Never execute live capital trades purely based on automated predictions without strict risk management."
        )
    )

    return bytes(pdf.output())


def generate_backtest_pdf(
    symbol: str,
    backtest_data: Dict,
    risk_reward_data: Optional[Dict] = None,
) -> bytes:
    """
    Generates a PDF document compiling the strategy backtest results,
    equity curves comparison metrics, trade log, and methodology notes.
    """
    s_m = backtest_data["strategy_metrics"]
    b_m = backtest_data["benchmark_metrics"]

    pdf = StockPDFReport(
        title_text=f"STRATEGY BACKTEST REPORT: {symbol}",
        subtitle_text=f"Period: {backtest_data.get('start_date')} to {backtest_data.get('end_date')} ({backtest_data.get('num_days')} days)",
    )
    pdf.add_page()

    # 1. Backtest Parameters
    pdf.draw_section_heading("1. BACKTEST CONFIGURATION")
    pdf.draw_key_value("Stock Symbol:", symbol, width=95)
    pdf.draw_key_value("Model Architecture:", backtest_data.get("model_name", "Linear Regression").upper(), width=95)
    pdf.ln(6)

    pdf.draw_key_value("Initial Portfolio Capital:", f"Rs. {backtest_data.get('initial_capital', 100000.0):,.0f}", width=95)
    pdf.draw_key_value("Transaction Cost / Friction:", f"{backtest_data.get('transaction_cost_pct', 0.10):.2f}% per trade", width=95)
    pdf.ln(8)

    # 2. Performance Comparison Table
    pdf.draw_section_heading("2. PERFORMANCE COMPARISON: ML STRATEGY VS BUY & HOLD")

    # Table Header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(70, 7, sanitize_text("Metric"), fill=True)
    pdf.cell(40, 7, sanitize_text("ML Strategy"), fill=True, align="R")
    pdf.cell(40, 7, sanitize_text("Buy & Hold"), fill=True, align="R")
    pdf.cell(40, 7, sanitize_text("Difference"), fill=True, align="R", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)

    rows = [
        ("Total Net Return", f"{s_m['total_return']:+.2f}%", f"{b_m['total_return']:+.2f}%", f"{s_m['total_return'] - b_m['total_return']:+.2f}%"),
        ("Annualized Return (CAGR)", f"{s_m['annualised_return']:+.2f}%", f"{b_m['annualised_return']:+.2f}%", f"{s_m['annualised_return'] - b_m['annualised_return']:+.2f}%"),
        ("Maximum Drawdown", f"{s_m['max_drawdown']:.2f}%", f"{b_m['max_drawdown']:.2f}%", f"{s_m['max_drawdown'] - b_m['max_drawdown']:+.2f}%"),
        ("Trades Executed", f"{s_m['trades']}", "1 (Buy & Hold)", f"{s_m['trades'] - 1:+d}"),
        ("Completed Round-Trips", f"{s_m['completed_trades']}", "1", f"{s_m['completed_trades'] - 1:+d}"),
        ("Trade Win Rate", f"{s_m['win_rate']:.1f}%", f"{b_m['win_rate']:.1f}% (up days)", f"{s_m['win_rate'] - b_m['win_rate']:+.1f}%"),
        ("Ending Portfolio Capital", f"Rs. {s_m['final_equity']:,.2f}", f"Rs. {b_m['final_equity']:,.2f}", f"Rs. {s_m['final_equity'] - b_m['final_equity']:+,.2f}"),
    ]

    fill = False
    for label, strat, bench, diff in rows:
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(70, 6, sanitize_text(label), fill=True)
        pdf.cell(40, 6, sanitize_text(strat), fill=True, align="R")
        pdf.cell(40, 6, sanitize_text(bench), fill=True, align="R")
        pdf.cell(40, 6, sanitize_text(diff), fill=True, align="R", new_x="LMARGIN", new_y="NEXT")
        fill = not fill

    pdf.ln(8)

    # 3. Trade Log Summary
    trade_events = backtest_data.get("trade_events", [])
    if trade_events:
        pdf.draw_section_heading("3. COMPLETED TRADE LOG (RECENT EXECUTIONS)")

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(226, 232, 240)
        pdf.cell(25, 6, sanitize_text("Entry Date"), fill=True)
        pdf.cell(25, 6, sanitize_text("Exit Date"), fill=True)
        pdf.cell(35, 6, sanitize_text("Entry Price"), fill=True, align="R")
        pdf.cell(35, 6, sanitize_text("Exit Price"), fill=True, align="R")
        pdf.cell(25, 6, sanitize_text("Holding"), fill=True, align="C")
        pdf.cell(25, 6, sanitize_text("Net Return"), fill=True, align="R")
        pdf.cell(20, 6, sanitize_text("Outcome"), fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 8)
        # Show top 8 recent trades
        for tr in trade_events[:8]:
            ret_pct = tr["trade_return"] * 100
            outcome = "Profit" if tr["is_win"] else "Loss"
            entry_d = str(tr["entry_date"])[:10]
            exit_d = str(tr["exit_date"])[:10]

            pdf.cell(25, 5, sanitize_text(entry_d))
            pdf.cell(25, 5, sanitize_text(exit_d))
            pdf.cell(35, 5, sanitize_text(f"Rs. {tr['entry_price']:,.2f}"), align="R")
            pdf.cell(35, 5, sanitize_text(f"Rs. {tr['exit_price']:,.2f}"), align="R")
            pdf.cell(25, 5, sanitize_text(f"{tr['holding_days']} days"), align="C")
            pdf.cell(25, 5, sanitize_text(f"{ret_pct:+.2f}%"), align="R")
            pdf.cell(20, 5, sanitize_text(outcome), align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(6)

    # 4. Critical Disclaimers
    pdf.draw_section_heading("4. CRITICAL METHODOLOGY & DATA LEAKAGE NOTICE")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(71, 85, 105)

    disclaimers_text = (
        "1. Training Data Contamination Warning: Models trained on historical periods (e.g. RELIANCE.NS) "
        "will exhibit in-sample lookahead bias if backtested on the training window. For realistic out-of-sample "
        "evaluation, test on other NSE tickers (TCS.NS, INFY.NS, HDFCBANK.NS).\n"
        "2. Friction Impact: Backtests reflect a simulated 0.10% friction per trade. High turnover strategies "
        "can accumulate substantial exchange transaction charges and STT.\n"
        "3. Past performance does not guarantee future results."
    )
    pdf.multi_cell(190, 4, sanitize_text(disclaimers_text))

    return bytes(pdf.output())
