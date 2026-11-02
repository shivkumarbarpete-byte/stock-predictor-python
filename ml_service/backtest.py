"""
ml_service/backtest.py
-----------------------
Simulation backtester for ML-guided long/flat trading strategies:
  - If the model's predicted return for tomorrow > 0: hold the stock tomorrow.
  - Else: stay in cash (0% return).
  - Configurable dynamic Stop-Loss and Take-Profit rules.
  - Advanced institutional trading metrics:
      * Sharpe Ratio (annualized, default 6% risk-free rate)
      * Sortino Ratio (annualized downside deviation)
      * Maximum Drawdown (MDD %)
      * Win / Loss Ratio (Avg Win / Avg Loss and Win Count / Loss Count)
      * Profit Factor (Gross Profits / Gross Losses)

Execution rules (No Look-Ahead):
  - At day t close, all technical indicators and model features are calculated.
  - Model generates predicted return for day t+1 based strictly on day t close data.
  - Position for day t+1 is initiated at day t close:
      signal(t+1) = 1 if predicted_return(t+1) > 0 else 0
  - If Stop-Loss or Take-Profit are enabled, intraday highs and lows of day t+1
    are evaluated against the trade's entry price:
      * Stop-Loss: Exit if Low(t+1) <= Entry * (1 - stop_loss_pct)
      * Take-Profit: Exit if High(t+1) >= Entry * (1 + take_profit_pct)
  - Trade execution / rebalance pays a configurable transaction cost (default 0.1%).
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

try:
    from ml_service.config import FEATURES
    from ml_service.stock_utils import fetch_stock_data, add_indicators, load_model
    from ml_service.risk_reward import compute_trading_metrics, compute_drawdown_series
except ImportError:
    from config import FEATURES
    from stock_utils import fetch_stock_data, add_indicators, load_model
    from risk_reward import compute_trading_metrics, compute_drawdown_series


def run_backtest(
    symbol: str = "RELIANCE.NS",
    model_name: str = "lr",
    period: str = "1y",
    transaction_cost: float = 0.001,
    initial_capital: float = 100000.0,
    stop_loss_pct: Optional[float] = None,    # e.g. 0.03 for 3% stop-loss
    take_profit_pct: Optional[float] = None,  # e.g. 0.06 for 6% take-profit
    risk_free_rate: float = 0.06,             # 6% per annum
) -> Dict[str, Any]:
    """
    Executes a backtest for the specified symbol and model over the given period
    with dynamic stop-loss, take-profit, and professional risk metrics.

    Parameters:
        symbol: NSE symbol ending in .NS (e.g. RELIANCE.NS)
        model_name: 'lr', 'rf', 'xgb', or 'lgbm'
        period: yfinance history period (e.g. '6mo', '1y', '2y')
        transaction_cost: Slippage & broker cost fraction per position change (0.001 = 0.1%)
        initial_capital: Starting portfolio value in INR (default 100,000)
        stop_loss_pct: Protective exit fraction (e.g. 0.03 for 3%), or None to disable
        take_profit_pct: Profit target fraction (e.g. 0.06 for 6%), or None to disable
        risk_free_rate: Annualized benchmark rate for Sharpe & Sortino (default 0.06)

    Returns:
        dict containing:
          - 'symbol', 'model_name', 'period', 'transaction_cost'
          - 'stop_loss_pct', 'take_profit_pct'
          - 'dates': pd.DatetimeIndex of backtest dates
          - 'daily_df': pd.DataFrame with daily signals, positions, returns, equity curves
          - 'strategy_metrics': dict of performance metrics (Sharpe, Sortino, MDD, Win/Loss, etc.)
          - 'benchmark_metrics': dict of buy & hold metrics
          - 'trade_events': list of individual trade entries & exits with exit reasons
    """
    # 1. Fetch data
    raw_df = fetch_stock_data(symbol, period=period)
    if raw_df.empty:
        raise ValueError(f"No historical price data returned for {symbol}.")

    # 2. Compute indicators and features
    df = add_indicators(raw_df, symbol=symbol)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Insufficient data points ({len(df)}) to run backtest.")

    # 3. Load model and predict next-day return
    model = load_model(model_name)
    X = df[FEATURES].values
    predicted_returns = model.predict(X)

    # 4. Construct backtest timeline without lookahead bias
    # At day t close: observe features df.iloc[t], model predicts return for t+1.
    # We evaluate for indices t = 0 to N-2
    n = len(df)
    dates = pd.to_datetime(df["Date"].iloc[1:]).reset_index(drop=True)
    opens = df["Open"].iloc[1:].values
    highs = df["High"].iloc[1:].values
    lows = df["Low"].iloc[1:].values
    closes = df["Close"].iloc[1:].values
    prev_closes = df["Close"].iloc[:-1].values
    asset_daily_returns = (closes / prev_closes) - 1.0

    # Model signal: 1 = Long tomorrow, 0 = Flat tomorrow
    signals = (predicted_returns[:-1] > 0).astype(int)

    # Clean percentages
    sl_pct = float(stop_loss_pct) if stop_loss_pct and stop_loss_pct > 0 else 0.0
    tp_pct = float(take_profit_pct) if take_profit_pct and take_profit_pct > 0 else 0.0

    # Simulation state variables
    positions = np.zeros(len(dates), dtype=int)
    strat_daily_returns = np.zeros(len(dates), dtype=float)
    trade_costs = np.zeros(len(dates), dtype=float)
    trades_count = 0
    trade_events: List[Dict[str, Any]] = []

    in_position = False
    trade_start_idx = None
    trade_entry_price = 0.0

    for i in range(len(dates)):
        day_date = dates[i]
        day_open = opens[i]
        day_high = highs[i]
        day_low = lows[i]
        day_close = closes[i]
        day_prev_close = prev_closes[i]
        model_sig = signals[i]

        # Case A: We are currently NOT in a position
        if not in_position:
            if model_sig == 1:
                # Enter Long at close of day t (which equals prev_close)
                in_position = True
                trade_start_idx = i
                trade_entry_price = day_prev_close
                positions[i] = 1
                trade_costs[i] = transaction_cost
                trades_count += 1

                # Check if stop-loss or take-profit are hit on the very first day held
                sl_hit = False
                tp_hit = False
                exit_price = day_close
                exit_reason = ""

                if sl_pct > 0:
                    sl_level = trade_entry_price * (1.0 - sl_pct)
                    if day_low <= sl_level:
                        sl_hit = True
                        exit_price = min(day_open, sl_level)
                        exit_reason = "Stop Loss"

                if not sl_hit and tp_pct > 0:
                    tp_level = trade_entry_price * (1.0 + tp_pct)
                    if day_high >= tp_level:
                        tp_hit = True
                        exit_price = tp_level
                        exit_reason = "Take Profit"

                if sl_hit or tp_hit:
                    # Exited intraday on the same day
                    in_position = False
                    positions[i] = 0
                    trade_costs[i] += transaction_cost
                    trades_count += 1
                    day_ret = (exit_price / trade_entry_price) - 1.0 - (2.0 * transaction_cost)
                    strat_daily_returns[i] = day_ret

                    trade_events.append({
                        "entry_date": df["Date"].iloc[trade_start_idx],
                        "exit_date": day_date,
                        "entry_price": round(float(trade_entry_price), 2),
                        "exit_price": round(float(exit_price), 2),
                        "holding_days": 1,
                        "trade_return": float(day_ret),
                        "is_win": bool(day_ret > 0),
                        "exit_reason": exit_reason,
                    })
                    trade_start_idx = None
                    trade_entry_price = 0.0
                else:
                    # Still held at day's close
                    strat_daily_returns[i] = (day_close / day_prev_close) - 1.0 - transaction_cost
            else:
                # Stay flat
                positions[i] = 0
                strat_daily_returns[i] = 0.0

        # Case B: We are ALREADY in a position from prior day
        else:
            positions[i] = 1
            sl_hit = False
            tp_hit = False
            exit_price = day_close
            exit_reason = ""

            # Check stop-loss
            if sl_pct > 0:
                sl_level = trade_entry_price * (1.0 - sl_pct)
                if day_low <= sl_level:
                    sl_hit = True
                    exit_price = min(day_open, sl_level)
                    exit_reason = "Stop Loss"

            # Check take-profit (if SL not hit)
            if not sl_hit and tp_pct > 0:
                tp_level = trade_entry_price * (1.0 + tp_pct)
                if day_high >= tp_level:
                    tp_hit = True
                    exit_price = tp_level
                    exit_reason = "Take Profit"

            if sl_hit or tp_hit:
                # Exit position due to price target rule
                in_position = False
                positions[i] = 0
                trade_costs[i] = transaction_cost
                trades_count += 1

                # Daily return from previous close to exit price
                day_ret = (exit_price / day_prev_close) - 1.0 - transaction_cost
                strat_daily_returns[i] = day_ret

                total_trade_ret = (exit_price / trade_entry_price) - 1.0 - (2.0 * transaction_cost)
                trade_events.append({
                    "entry_date": df["Date"].iloc[trade_start_idx],
                    "exit_date": day_date,
                    "entry_price": round(float(trade_entry_price), 2),
                    "exit_price": round(float(exit_price), 2),
                    "holding_days": max(i - trade_start_idx + 1, 1),
                    "trade_return": float(total_trade_ret),
                    "is_win": bool(total_trade_ret > 0),
                    "exit_reason": exit_reason,
                })
                trade_start_idx = None
                trade_entry_price = 0.0

            elif model_sig == 0:
                # Exit position because model turned bearish/neutral (signal flip)
                in_position = False
                positions[i] = 0
                trade_costs[i] = transaction_cost
                trades_count += 1

                day_ret = (day_close / day_prev_close) - 1.0 - transaction_cost
                strat_daily_returns[i] = day_ret

                total_trade_ret = (day_close / trade_entry_price) - 1.0 - (2.0 * transaction_cost)
                trade_events.append({
                    "entry_date": df["Date"].iloc[trade_start_idx],
                    "exit_date": day_date,
                    "entry_price": round(float(trade_entry_price), 2),
                    "exit_price": round(float(day_close), 2),
                    "holding_days": max(i - trade_start_idx + 1, 1),
                    "trade_return": float(total_trade_ret),
                    "is_win": bool(total_trade_ret > 0),
                    "exit_reason": "Signal Flip",
                })
                trade_start_idx = None
                trade_entry_price = 0.0
            else:
                # Model remains bullish; hold position
                strat_daily_returns[i] = (day_close / day_prev_close) - 1.0

    # If still in trade at the end of backtest, close out for reporting
    if in_position and trade_start_idx is not None:
        exit_price = closes[-1]
        trade_ret = (exit_price / trade_entry_price) - 1.0 - transaction_cost
        trade_events.append({
            "entry_date": df["Date"].iloc[trade_start_idx],
            "exit_date": dates.iloc[-1],
            "entry_price": round(float(trade_entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "holding_days": max(len(positions) - trade_start_idx, 1),
            "trade_return": float(trade_ret),
            "is_win": bool(trade_ret > 0),
            "exit_reason": "End of Period",
        })

    # Strategy Equity curve
    strat_cumprod = np.cumprod(1.0 + strat_daily_returns)
    strat_equity = initial_capital * strat_cumprod

    # Buy and hold benchmark: enters at start (pays 1 trade cost) and holds
    bh_costs = np.zeros(len(asset_daily_returns))
    bh_costs[0] = transaction_cost
    bh_daily_returns = asset_daily_returns - bh_costs
    bh_cumprod = np.cumprod(1.0 + bh_daily_returns)
    bh_equity = initial_capital * bh_cumprod

    # Compile daily timeline DataFrame
    daily_df = pd.DataFrame({
        "Date": dates,
        "Close": closes,
        "Predicted_Return": predicted_returns[:-1],
        "Position": positions,
        "Asset_Return": asset_daily_returns,
        "Strategy_Return": strat_daily_returns,
        "Strategy_Equity": strat_equity,
        "Benchmark_Equity": bh_equity,
    })

    # Drawdown series
    strat_dd = compute_drawdown_series(pd.Series(strat_equity))
    bh_dd = compute_drawdown_series(pd.Series(bh_equity))
    daily_df["Strategy_Drawdown"] = strat_dd
    daily_df["Benchmark_Drawdown"] = bh_dd

    # Compute institutional performance & risk metrics
    strat_metrics_raw = compute_trading_metrics(
        daily_returns=pd.Series(strat_daily_returns),
        equity_series=pd.Series(strat_equity),
        trade_events=trade_events,
        risk_free_rate=risk_free_rate,
    )

    bh_benchmark_events = [{
        "trade_return": (bh_equity[-1] / initial_capital) - 1.0,
        "is_win": (bh_equity[-1] > initial_capital),
    }]
    bh_metrics_raw = compute_trading_metrics(
        daily_returns=pd.Series(bh_daily_returns),
        equity_series=pd.Series(bh_equity),
        trade_events=bh_benchmark_events,
        risk_free_rate=risk_free_rate,
    )

    num_days = len(daily_df)
    years = max(num_days / 252.0, 1.0 / 252.0)

    # Strategy returns & CAGR
    strat_total_ret = (strat_equity[-1] / initial_capital) - 1.0
    strat_cagr = ((1.0 + strat_total_ret) ** (1.0 / years)) - 1.0 if (1.0 + strat_total_ret) > 0 else -1.0

    # Benchmark returns & CAGR
    bh_total_ret = (bh_equity[-1] / initial_capital) - 1.0
    bh_cagr = ((1.0 + bh_total_ret) ** (1.0 / years)) - 1.0 if (1.0 + bh_total_ret) > 0 else -1.0

    # Win rate calculation
    if len(trade_events) > 0:
        win_count = sum(1 for t in trade_events if t["is_win"])
        win_rate = (win_count / len(trade_events)) * 100.0
    else:
        active_returns = strat_daily_returns[positions == 1]
        win_rate = (np.mean(active_returns > 0) * 100.0) if len(active_returns) > 0 else 0.0

    strategy_metrics = {
        "total_return": round(float(strat_total_ret * 100.0), 2),
        "annualised_return": round(float(strat_cagr * 100.0), 2),
        "max_drawdown": strat_metrics_raw["max_drawdown"],
        "sharpe_ratio": strat_metrics_raw["sharpe_ratio"],
        "sortino_ratio": strat_metrics_raw["sortino_ratio"],
        "win_loss_ratio": strat_metrics_raw["win_loss_ratio"],
        "profit_factor": strat_metrics_raw["profit_factor"],
        "avg_win_pct": strat_metrics_raw["avg_win_pct"],
        "avg_loss_pct": strat_metrics_raw["avg_loss_pct"],
        "win_count": strat_metrics_raw["win_count"],
        "loss_count": strat_metrics_raw["loss_count"],
        "trades": trades_count,
        "completed_trades": len(trade_events),
        "win_rate": round(float(win_rate), 1),
        "final_equity": round(float(strat_equity[-1]), 2),
    }

    benchmark_metrics = {
        "total_return": round(float(bh_total_ret * 100.0), 2),
        "annualised_return": round(float(bh_cagr * 100.0), 2),
        "max_drawdown": bh_metrics_raw["max_drawdown"],
        "sharpe_ratio": bh_metrics_raw["sharpe_ratio"],
        "sortino_ratio": bh_metrics_raw["sortino_ratio"],
        "win_loss_ratio": bh_metrics_raw["win_loss_ratio"],
        "profit_factor": bh_metrics_raw["profit_factor"],
        "avg_win_pct": bh_metrics_raw["avg_win_pct"],
        "avg_loss_pct": bh_metrics_raw["avg_loss_pct"],
        "win_count": bh_metrics_raw["win_count"],
        "loss_count": bh_metrics_raw["loss_count"],
        "trades": 1,
        "completed_trades": 1,
        "win_rate": round(float((np.mean(asset_daily_returns > 0) * 100.0)), 1),
        "final_equity": round(float(bh_equity[-1]), 2),
    }

    return {
        "symbol": symbol.upper(),
        "model_name": model_name,
        "period": period,
        "transaction_cost_pct": transaction_cost * 100.0,
        "initial_capital": initial_capital,
        "stop_loss_pct": round(sl_pct * 100.0, 2) if sl_pct > 0 else None,
        "take_profit_pct": round(tp_pct * 100.0, 2) if tp_pct > 0 else None,
        "risk_free_rate_pct": round(risk_free_rate * 100.0, 1),
        "num_days": num_days,
        "start_date": daily_df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": daily_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "daily_df": daily_df,
        "trade_events": trade_events,
    }


if __name__ == "__main__":
    res = run_backtest(
        symbol="RELIANCE.NS",
        model_name="lr",
        period="1y",
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
    )
    print(f"Backtest {res['symbol']} ({res['start_date']} to {res['end_date']})")
    print("Strategy metrics:", res["strategy_metrics"])
    print("Benchmark metrics:", res["benchmark_metrics"])
    print("Completed trades count:", len(res["trade_events"]))
