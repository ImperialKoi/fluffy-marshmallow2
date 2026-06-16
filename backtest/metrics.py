"""
Performance metrics.

These are the numbers that tell you whether a strategy is actually any good. The
ones that matter most for a sanity check:
  * Total / annualised return vs the buy & hold benchmark (did you beat doing
    nothing? most strategies don't).
  * Sharpe & Sortino (return per unit of risk).
  * Max drawdown (the worst peak-to-trough loss — what your stomach has to take).
  * Win rate & trade count (is there enough evidence, or did 3 lucky trades carry it?).
"""

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


def _cagr(curve: pd.Series, periods_per_year: int) -> float:
    years = len(curve) / periods_per_year
    if years <= 0 or curve.iloc[0] <= 0:
        return 0.0
    return (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1


def _max_drawdown(curve: pd.Series) -> float:
    running_peak = curve.cummax()
    drawdown = (curve - running_peak) / running_peak
    return drawdown.min()


def _sharpe(returns: pd.Series, periods_per_year: int, rf: float) -> float:
    excess = returns - rf / periods_per_year
    sd = excess.std()
    if sd == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / sd


def _sortino(returns: pd.Series, periods_per_year: int, rf: float) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / downside


def compute_metrics(result: BacktestResult, periods_per_year: int = 252,
                    rf: float = 0.0) -> dict:
    curve = result.equity_curve
    returns = result.returns
    bench = result.benchmark_curve

    # Trade-level win rate: pair each closing trade against its entry cost basis.
    realised = _trade_pnl(result.trades)
    wins = [p for p in realised if p > 0]

    total_fees = sum(t.cost for t in result.trades)
    max_dd = _max_drawdown(curve)
    cagr = _cagr(curve, periods_per_year)

    return {
        "strategy": result.strategy_name,
        "symbol": result.symbol,
        "start": curve.index[0].date().isoformat(),
        "end": curve.index[-1].date().isoformat(),
        "bars": len(curve),
        "final_equity": curve.iloc[-1],
        "total_return": curve.iloc[-1] / curve.iloc[0] - 1,
        "cagr": cagr,
        "ann_volatility": returns.std() * np.sqrt(periods_per_year),
        "sharpe": _sharpe(returns, periods_per_year, rf),
        "sortino": _sortino(returns, periods_per_year, rf),
        "max_drawdown": max_dd,
        "calmar": (cagr / abs(max_dd)) if max_dd != 0 else 0.0,
        "num_trades": len(result.trades),
        "round_trips": len(realised),
        "win_rate": (len(wins) / len(realised)) if realised else 0.0,
        "total_fees": total_fees,
        "exposure": (returns != 0).mean(),
        # benchmark
        "benchmark_total_return": bench.iloc[-1] / bench.iloc[0] - 1,
        "benchmark_cagr": _cagr(bench, periods_per_year),
        "benchmark_max_drawdown": _max_drawdown(bench),
    }


def _trade_pnl(trades):
    """Reconstruct realised P&L per round trip (FIFO, long-only friendly)."""
    pnl = []
    open_price = None
    open_shares = 0
    for tr in trades:
        signed = tr.shares if tr.side == "BUY" else -tr.shares
        if open_shares == 0:
            open_price, open_shares = tr.price, signed
        elif np.sign(signed) != np.sign(open_shares):
            # closing (fully, for the simple long-only/short-only demo case)
            direction = np.sign(open_shares)
            pnl.append(direction * (tr.price - open_price) * abs(open_shares))
            open_shares += signed
            if open_shares != 0:
                open_price = tr.price
        else:
            open_shares += signed
    return pnl


def format_report(metrics: dict) -> str:
    m = metrics
    pct = lambda x: f"{x * 100:,.2f}%"
    lines = [
        f"  Strategy        : {m['strategy']}",
        f"  Symbol / period : {m['symbol']}  {m['start']} -> {m['end']}  ({m['bars']} bars)",
        "  " + "-" * 52,
        f"  Final equity    : ${m['final_equity']:,.2f}",
        f"  Total return    : {pct(m['total_return'])}   (buy&hold {pct(m['benchmark_total_return'])})",
        f"  CAGR            : {pct(m['cagr'])}   (buy&hold {pct(m['benchmark_cagr'])})",
        f"  Ann. volatility : {pct(m['ann_volatility'])}",
        f"  Sharpe / Sortino: {m['sharpe']:.2f} / {m['sortino']:.2f}",
        f"  Max drawdown    : {pct(m['max_drawdown'])}   (buy&hold {pct(m['benchmark_max_drawdown'])})",
        f"  Calmar          : {m['calmar']:.2f}",
        f"  Round trips     : {m['round_trips']}   win rate {pct(m['win_rate'])}",
        f"  Total fees paid : ${m['total_fees']:,.2f}",
        f"  Time in market  : {pct(m['exposure'])}",
    ]
    return "\n".join(lines)
