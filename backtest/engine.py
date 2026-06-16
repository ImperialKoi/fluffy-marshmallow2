"""
Event-driven backtest engine.

Design goals, in priority order: (1) correctness, (2) realism, (3) clarity.
Speed is a non-issue for daily data over a few years.

The honest-backtest contract:
  * A signal is computed from the CLOSE of bar i (information you really had).
  * The resulting order fills at the OPEN of bar i+1 (the next tradable price).
  * Slippage makes your fill worse than the quoted open; commission is charged on
    notional. Both always work against you.
  * Stop-loss / take-profit are checked against the intraday high/low of each bar.
  * The portfolio kill switch can halt trading mid-run.

This mirrors how a live bot would actually behave, so results transfer instead of
surprising you.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from strategies.base import Strategy
from risk.manager import RiskManager


@dataclass
class Trade:
    date: pd.Timestamp
    side: str          # "BUY" / "SELL"
    shares: int
    price: float       # fill price after slippage
    reason: str        # "signal" / "stop" / "take_profit" / "kill_switch"
    cost: float        # commission paid


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    trades: List[Trade]
    benchmark_curve: pd.Series
    strategy_name: str
    symbol: str


def _fill_price(quote: float, side: int, slippage_bps: float) -> float:
    """Apply slippage against the trader: buys fill higher, sells fill lower."""
    slip = slippage_bps / 10_000.0
    return quote * (1 + slip) if side > 0 else quote * (1 - slip)


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskManager,
                 initial_cash: float, commission_bps: float, slippage_bps: float,
                 symbol: str) -> BacktestResult:

    df = strategy.prepare(df)
    n = len(df)
    if n < 2:
        raise ValueError("Need at least 2 bars to run a backtest.")

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    dates = df.index

    cash = initial_cash
    shares = 0
    side = 0                 # current position direction: -1/0/+1
    entry_price = 0.0
    stop = tp = None
    commission = commission_bps / 10_000.0

    equity = np.empty(n)
    trades: List[Trade] = []

    # Signal decided at the close of the *previous* bar drives this bar's open.
    pending_target = strategy.signal(df, 0)
    equity[0] = cash         # day 0: flat, just observing

    for t in range(1, n):
        # --- 1. Execute the pending order at this bar's OPEN -------------------
        halted = risk.halted
        desired_side = 0 if halted else int(np.sign(pending_target))
        if desired_side != side:
            mark_equity = cash + shares * opens[t]
            target_sh = risk.target_shares(desired_side, mark_equity, opens[t])
            delta = target_sh - shares
            if delta != 0:
                trade_side = 1 if delta > 0 else -1
                fill = _fill_price(opens[t], trade_side, slippage_bps)
                notional = abs(delta) * fill
                fee = notional * commission
                cash -= delta * fill          # buying delta>0 reduces cash
                cash -= fee
                shares = target_sh
                side = int(np.sign(shares))
                trades.append(Trade(dates[t], "BUY" if delta > 0 else "SELL",
                                    abs(delta), fill,
                                    "kill_switch" if halted else "signal", fee))
                if side != 0:
                    entry_price = fill
                    stop, tp = risk.stop_levels(entry_price, side)
                else:
                    entry_price, stop, tp = 0.0, None, None

        # --- 2. Intraday stop-loss / take-profit ------------------------------
        if side != 0:
            triggered, exit_price = risk.check_stops(side, highs[t], lows[t], stop, tp)
            if triggered:
                fill = _fill_price(exit_price, -side, slippage_bps)
                notional = abs(shares) * fill
                fee = notional * commission
                reason = "stop" if (
                    (side > 0 and exit_price == stop) or (side < 0 and exit_price == stop)
                ) else "take_profit"
                cash += shares * fill         # closing the position
                cash -= fee
                trades.append(Trade(dates[t], "SELL" if side > 0 else "BUY",
                                    abs(shares), fill, reason, fee))
                shares, side, entry_price, stop, tp = 0, 0, 0.0, None, None

        # --- 3. Mark to market at the CLOSE -----------------------------------
        equity[t] = cash + shares * closes[t]

        # --- 4. Drawdown kill switch, then next signal ------------------------
        risk.update_drawdown(equity[t])
        pending_target = strategy.signal(df, t)

    equity_curve = pd.Series(equity, index=dates, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)

    # Buy & hold benchmark on the same series (bought at first available open).
    bench_shares = initial_cash / opens[0]
    benchmark_curve = pd.Series(bench_shares * closes, index=dates, name="benchmark")
    benchmark_curve.iloc[0] = initial_cash

    return BacktestResult(equity_curve, returns, trades, benchmark_curve,
                          strategy.name, symbol)
