"""
ml_service/backtest.py
-----------------------
Vectorized and event-aware backtester for a simple long/flat strategy:
  - If the model's predicted return for tomorrow > 0: hold the stock tomorrow.
  - Else: stay in cash (0% return).

Execution rules (No Look-Ahead):
  - At day t close, all technical indicators and model features are calculated.
  - Model generates predicted return for day t+1 based strictly on day t close data.
  - Position for day t+1 is determined at day t close:
      position(t+1) = 1 if predicted_return(t+1) > 0 else 0
  - Trade execution / rebalance occurs at the transition, paying a configurable
    transaction cost (default 0.1% per trade).
  - Day t+1 realized return is Close(t+1) / Close(t) - 1.

Metrics reported (Strategy vs Buy-and-Hold):
  - Total Return (%)
  - Annualised Return (CAGR %)
  - Maximum Drawdown (%)
  - Number of Trades (position changes)
  - Win Rate (% of winning closed trades / days)
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    from ml_service.config import FEATURES
    from ml_service.stock_utils import fetch_stock_data, add_indicators, load_model
except ImportError:
    from config import FEATURES
    from stock_utils import fetch_stock_data, add_indicators, load_model


def compute_drawdown_series(equity_series: pd.Series) -> pd.Series:
    """Computes percentage drawdown from running peak for an equity series."""
    running_max = equity_series.cummax()
    return (equity_series - running_max) / running_max * 100.0


def run_backtest(
    symbol: str = "RELIANCE.NS",
    model_name: str = "lr",
    period: str = "1y",
    transaction_cost: float = 0.001,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """
    Executes a backtest for the specified symbol and model over the given period.

    Parameters:
        symbol: NSE symbol ending in .NS (e.g. RELIANCE.NS)
        model_name: 'lr' (Linear Regression) or 'rf' (Random Forest)
        period: yfinance history period (e.g. '6mo', '1y', '2y')
        transaction_cost: Slippage & broker cost fraction per position change (0.001 = 0.1%)
        initial_capital: Starting portfolio value in INR (default 100,000)

    Returns:
        dict containing:
          - 'symbol', 'model_name', 'period', 'transaction_cost'
          - 'dates': pd.DatetimeIndex of backtest dates
          - 'daily_df': pd.DataFrame with daily signals, positions, returns, equity curves
          - 'strategy_metrics': dict of performance metrics
          - 'benchmark_metrics': dict of buy & hold metrics
          - 'trade_log': list of individual trade entries & exits
    """
    # 1. Fetch data
    raw_df = fetch_stock_data(symbol, period=period)
    if raw_df.empty:
        raise ValueError(f"No historical price data returned for {symbol}.")

    # 2. Compute indicators and features
    df = add_indicators(raw_df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Insufficient data points ({len(df)}) to run backtest.")

    # 3. Load model and predict next-day return
    model = load_model(model_name)
    X = df[FEATURES].values
    predicted_returns = model.predict(X)

    # 4. Construct backtest timeline without lookahead bias
    # At day t close: we observe features df.iloc[t] and obtain predicted_returns[t].
    # This determines the target position for the NEXT trading day (t -> t+1).
    # Actual asset return from day t to day t+1 is Close[t+1] / Close[t] - 1.
    
    # We evaluate for indices t = 0 to N-2
    n = len(df)
    dates = pd.to_datetime(df["Date"].iloc[1:]).reset_index(drop=True)
    actual_returns = (df["Close"].iloc[1:].values / df["Close"].iloc[:-1].values) - 1.0
    signals = (predicted_returns[:-1] > 0).astype(int)  # 1 = Long, 0 = Flat (Cash)

    # Calculate position changes and transaction costs
    positions = signals
    trades_count = 0
    trade_costs = np.zeros(len(positions))
    trade_events = []

    current_pos = 0
    trade_start_idx = None
    trade_entry_price = None

    for i in range(len(positions)):
        new_pos = positions[i]
        if new_pos != current_pos:
            trade_costs[i] = transaction_cost
            trades_count += 1
            
            # Position change event
            if current_pos == 0 and new_pos == 1:
                trade_start_idx = i
                trade_entry_price = df["Close"].iloc[i]  # entered at close of day t
            elif current_pos == 1 and new_pos == 0:
                exit_price = df["Close"].iloc[i]
                trade_ret = (exit_price / trade_entry_price) - 1.0 - (2 * transaction_cost)
                trade_events.append({
                    "entry_date": df["Date"].iloc[trade_start_idx],
                    "exit_date": df["Date"].iloc[i],
                    "entry_price": trade_entry_price,
                    "exit_price": exit_price,
                    "holding_days": i - trade_start_idx,
                    "trade_return": trade_ret,
                    "is_win": trade_ret > 0,
                })
                trade_start_idx = None
                trade_entry_price = None

        current_pos = new_pos

    # If still in trade at the end of backtest, close out for reporting
    if current_pos == 1 and trade_start_idx is not None:
        exit_price = df["Close"].iloc[-1]
        trade_ret = (exit_price / trade_entry_price) - 1.0 - transaction_cost
        trade_events.append({
            "entry_date": df["Date"].iloc[trade_start_idx],
            "exit_date": df["Date"].iloc[-1],
            "entry_price": trade_entry_price,
            "exit_price": exit_price,
            "holding_days": len(positions) - trade_start_idx,
            "trade_return": trade_ret,
            "is_win": trade_ret > 0,
        })

    # Strategy daily net return:
    # If position == 1, earn asset_return - cost. If position == 0, earn 0 - cost.
    strat_daily_returns = (positions * actual_returns) - trade_costs

    # Equity curves
    strat_cumprod = np.cumprod(1.0 + strat_daily_returns)
    strat_equity = initial_capital * strat_cumprod

    # Buy and hold benchmark: enters at start (pays 1 trade cost) and holds
    bh_costs = np.zeros(len(actual_returns))
    bh_costs[0] = transaction_cost
    bh_daily_returns = actual_returns - bh_costs
    bh_cumprod = np.cumprod(1.0 + bh_daily_returns)
    bh_equity = initial_capital * bh_cumprod

    # Compile daily timeline DataFrame
    daily_df = pd.DataFrame({
        "Date": dates,
        "Close": df["Close"].iloc[1:].values,
        "Predicted_Return": predicted_returns[:-1],
        "Position": positions,
        "Asset_Return": actual_returns,
        "Strategy_Return": strat_daily_returns,
        "Strategy_Equity": strat_equity,
        "Benchmark_Equity": bh_equity,
    })

    # Drawdowns
    strat_dd = compute_drawdown_series(pd.Series(strat_equity))
    bh_dd = compute_drawdown_series(pd.Series(bh_equity))
    daily_df["Strategy_Drawdown"] = strat_dd
    daily_df["Benchmark_Drawdown"] = bh_dd

    # Compute key metrics
    num_days = len(daily_df)
    years = max(num_days / 252.0, 1.0 / 252.0)

    # Strategy metrics
    strat_total_ret = (strat_equity[-1] / initial_capital) - 1.0
    strat_cagr = ((1.0 + strat_total_ret) ** (1.0 / years)) - 1.0 if (1.0 + strat_total_ret) > 0 else -1.0
    strat_max_dd = float(strat_dd.min())

    # Benchmark metrics
    bh_total_ret = (bh_equity[-1] / initial_capital) - 1.0
    bh_cagr = ((1.0 + bh_total_ret) ** (1.0 / years)) - 1.0 if (1.0 + bh_total_ret) > 0 else -1.0
    bh_max_dd = float(bh_dd.min())

    # Win rate calculation
    if len(trade_events) > 0:
        win_count = sum(1 for t in trade_events if t["is_win"])
        win_rate = (win_count / len(trade_events)) * 100.0
    else:
        # Fallback to active daily win rate if zero discrete trade exits occurred
        active_returns = strat_daily_returns[positions == 1]
        win_rate = (np.mean(active_returns > 0) * 100.0) if len(active_returns) > 0 else 0.0

    strategy_metrics = {
        "total_return": round(float(strat_total_ret * 100.0), 2),
        "annualised_return": round(float(strat_cagr * 100.0), 2),
        "max_drawdown": round(float(strat_max_dd), 2),
        "trades": trades_count,
        "completed_trades": len(trade_events),
        "win_rate": round(float(win_rate), 1),
        "final_equity": round(float(strat_equity[-1]), 2),
    }

    benchmark_metrics = {
        "total_return": round(float(bh_total_ret * 100.0), 2),
        "annualised_return": round(float(bh_cagr * 100.0), 2),
        "max_drawdown": round(float(bh_max_dd), 2),
        "trades": 1,
        "completed_trades": 1,
        "win_rate": round(float((np.mean(actual_returns > 0) * 100.0)), 1),
        "final_equity": round(float(bh_equity[-1]), 2),
    }

    return {
        "symbol": symbol.upper(),
        "model_name": model_name,
        "period": period,
        "transaction_cost_pct": transaction_cost * 100.0,
        "initial_capital": initial_capital,
        "num_days": num_days,
        "start_date": daily_df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": daily_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "daily_df": daily_df,
        "trade_events": trade_events,
    }


if __name__ == "__main__":
    res = run_backtest(symbol="RELIANCE.NS", model_name="lr", period="1y")
    print(f"Backtest {res['symbol']} ({res['start_date']} to {res['end_date']})")
    print("Strategy:", res["strategy_metrics"])
    print("Benchmark:", res["benchmark_metrics"])
