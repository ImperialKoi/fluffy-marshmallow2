"""
Correctness tests for the engine. Run with:  python tests/test_engine.py

These use a tiny hand-built dataset so the expected fills can be computed by
hand, which is the only real way to trust a backtester. The headline check is
the no-lookahead property: a signal computed on the close of bar i must fill at
the OPEN of bar i+1, never sooner.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from strategies.base import Strategy
from risk.manager import RiskManager
from backtest.engine import run_backtest


def _toy_df():
    # 4 days. Opens differ from closes so we can tell which price was used.
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    return pd.DataFrame({
        "open":  [100.0, 110.0, 120.0, 130.0],
        "high":  [105.0, 115.0, 125.0, 135.0],
        "low":   [ 99.0, 109.0, 119.0, 129.0],
        "close": [102.0, 112.0, 122.0, 132.0],
        "volume":[1_000, 1_000, 1_000, 1_000],
    }, index=idx)


class GoLongDay0(Strategy):
    """Signal long on the very first close; stay long forever."""
    name = "test-go-long"
    def signal(self, df, i):
        return 1


def test_fills_at_next_open():
    df = _toy_df()
    risk = RiskManager(position_fraction=1.0, stop_loss_pct=None,
                       take_profit_pct=None, max_drawdown_kill=None)
    res = run_backtest(df, GoLongDay0(), risk, initial_cash=10_000.0,
                       commission_bps=0.0, slippage_bps=0.0, symbol="TOY")

    assert len(res.trades) == 1, "should open exactly one position"
    trade = res.trades[0]
    # Signal decided on close of bar 0 -> must fill at OPEN of bar 1 = 110.0,
    # NOT bar 0's open (100) or close (102). That is the no-lookahead guarantee.
    assert trade.date == df.index[1], f"fill date wrong: {trade.date}"
    assert trade.price == 110.0, f"expected next-open fill 110.0, got {trade.price}"
    shares = int(10_000 // 110.0)            # 90 shares
    assert trade.shares == shares
    # Final equity = cash left + shares * last close
    cash_left = 10_000.0 - shares * 110.0
    expected = cash_left + shares * 132.0
    assert abs(res.equity_curve.iloc[-1] - expected) < 1e-6
    print("PASS: fills at next open, no lookahead, equity exact")


def test_slippage_hurts_the_trader():
    df = _toy_df()
    risk = RiskManager(position_fraction=1.0, stop_loss_pct=None,
                       take_profit_pct=None, max_drawdown_kill=None)
    res = run_backtest(df, GoLongDay0(), risk, initial_cash=10_000.0,
                       commission_bps=0.0, slippage_bps=50.0, symbol="TOY")
    # 50 bps slippage on a buy => fill above the 110 open.
    assert res.trades[0].price > 110.0
    assert abs(res.trades[0].price - 110.0 * 1.005) < 1e-9
    print("PASS: slippage makes buy fills worse")


def test_stop_loss_triggers():
    # Build a series that gaps down so an 8% stop must fire.
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    df = pd.DataFrame({
        "open":  [100.0, 100.0,  90.0],
        "high":  [101.0, 101.0,  91.0],
        "low":   [ 99.0,  99.0,  80.0],   # day 2 low 80 -> below 8% stop (~92)
        "close": [100.0, 100.0,  85.0],
        "volume":[1_000, 1_000, 1_000],
    }, index=idx)
    risk = RiskManager(position_fraction=1.0, stop_loss_pct=0.08,
                       take_profit_pct=None, max_drawdown_kill=None)
    res = run_backtest(df, GoLongDay0(), risk, initial_cash=10_000.0,
                       commission_bps=0.0, slippage_bps=0.0, symbol="TOY")
    reasons = [t.reason for t in res.trades]
    assert "stop" in reasons, f"stop loss should have triggered, got {reasons}"
    print("PASS: stop loss triggers on intraday breach")


if __name__ == "__main__":
    test_fills_at_next_open()
    test_slippage_hurts_the_trader()
    test_stop_loss_triggers()
    print("\nAll engine correctness tests passed.")
