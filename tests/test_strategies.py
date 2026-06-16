"""
Library-wide correctness tests for every registered strategy. Run with:

    python tests/test_strategies.py

Two properties are checked for EACH strategy in strategies/registry.py:

  1. No lookahead. The decision a strategy makes at bar i must not change when
     future bars are appended. We verify this directly: prepare()+signal() on the
     full series at bar i must equal prepare()+signal() on the series TRUNCATED at
     bar i. If any indicator peeked ahead (center=True, shift(-k), whole-series
     stats, ...) the truncated frame would disagree. This is the single most
     important invariant in CLAUDE.md §3.

  2. Runs end-to-end. Each strategy must complete a full backtest through the real
     engine on the bundled dataset and produce a finite equity curve.
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np

import config
from data import loader
from risk.manager import RiskManager
from backtest.engine import run_backtest
from strategies.registry import REGISTRY, build, names

SYMBOL = "AAPL"
CHECK_INDICES = [260, 500, 800, 1000, 1258]


def _data():
    return loader.load_csv(config.CACHE_CSV, SYMBOL)


def test_no_lookahead():
    df = _data()
    n = len(df)
    failures = []
    for nm in names():
        strat = build(nm)
        full = strat.prepare(df.copy())
        for i in CHECK_INDICES:
            if i >= n:
                continue
            trunc = strat.prepare(df.iloc[:i + 1].copy())
            s_full = int(strat.signal(full, i))
            s_trunc = int(strat.signal(trunc, i))
            if s_full != s_trunc:
                failures.append(f"{nm} @ i={i}: full={s_full} trunc={s_trunc}")
    assert not failures, "LOOKAHEAD DETECTED:\n  " + "\n  ".join(failures)
    print(f"PASS: no-lookahead holds for all {len(names())} strategies "
          f"at indices {CHECK_INDICES}")


def test_runs_end_to_end():
    df = _data()
    failures = []
    for nm in names():
        try:
            strat = build(nm)
            risk = RiskManager(position_fraction=0.95, stop_loss_pct=0.08,
                               take_profit_pct=None, max_drawdown_kill=0.25)
            res = run_backtest(df.copy(), strat, risk, initial_cash=100_000.0,
                               commission_bps=0.0, slippage_bps=5.0, symbol=SYMBOL)
            eq = res.equity_curve.iloc[-1]
            if not np.isfinite(eq):
                failures.append(f"{nm}: non-finite final equity {eq}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{nm}: raised {e!r}")
    assert not failures, "ENGINE RUN FAILURES:\n  " + "\n  ".join(failures)
    print(f"PASS: all {len(names())} strategies run end-to-end through the engine")


def test_signals_are_valid():
    """Every signal must be one of -1, 0, +1."""
    df = _data()
    failures = []
    for nm in names():
        strat = build(nm)
        prepared = strat.prepare(df.copy())
        for i in CHECK_INDICES:
            if i >= len(df):
                continue
            s = strat.signal(prepared, i)
            if s not in (-1, 0, 1):
                failures.append(f"{nm} @ i={i}: signal {s!r} not in (-1,0,1)")
    assert not failures, "INVALID SIGNALS:\n  " + "\n  ".join(failures)
    print(f"PASS: all signals are in {{-1, 0, +1}}")


if __name__ == "__main__":
    test_no_lookahead()
    test_signals_are_valid()
    test_runs_end_to_end()
    print("\nAll strategy-library tests passed.")
