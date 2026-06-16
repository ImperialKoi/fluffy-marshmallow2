"""
Train/test splits and walk-forward analysis — the leakage-proof core.

THE CARDINAL RULE: out-of-sample (test) data must never influence a train-period
choice. This module is built so that property holds *structurally*:

  * The engine produces, for a given (strategy, params), one full-history equity
    curve. Equity at bar i depends only on bars <= i (the engine has no lookahead).
  * Therefore the metric over a TRAIN window depends only on bars up to that
    window's end. Parameter selection reads train-window metrics ONLY, so it is
    provably independent of anything in (or after) the test window.
  * `walk_forward` selects per fold on the train window, then scores the next,
    unseen test window exactly once, and stitches the OOS segments together.

tests/test_validation_leakage.py corrupts every bar after a fold's train end and
asserts the fold's train metrics and parameter choice are byte-for-byte identical
(while OOS results change) — a mechanical proof that no future data leaks back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from backtest.metrics import compute_metrics
from . import config as vcfg


def to_ts(x):
    return pd.Timestamp(x)


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def slice_result(res: BacktestResult, start, end) -> BacktestResult:
    """A BacktestResult restricted to [start, end]. Returns are recomputed within
    the window (first day 0) so metrics are window-relative, not contaminated by the
    jump from the prior segment."""
    start, end = to_ts(start), to_ts(end)
    eq = res.equity_curve.loc[start:end]
    bench = res.benchmark_curve.loc[start:end]
    rets = eq.pct_change().fillna(0.0)
    trades = [t for t in res.trades if start <= t.date <= end]
    return BacktestResult(eq, rets, trades, bench, res.strategy_name, res.symbol)


def window_metrics(res: BacktestResult, start, end) -> dict | None:
    sub = slice_result(res, start, end)
    if len(sub.equity_curve) < 2:
        return None
    return compute_metrics(sub, vcfg.PERIODS_PER_YEAR, vcfg.RISK_FREE)


def window_returns(res: BacktestResult, start, end) -> pd.Series:
    start, end = to_ts(start), to_ts(end)
    eq = res.equity_curve.loc[start:end]
    return eq.pct_change().fillna(0.0)


# --------------------------------------------------------------------------- #
# parameter selection — TRAIN ONLY
# --------------------------------------------------------------------------- #
def select_on_train(results: dict, grid: list[dict], train_start, train_end,
                    min_train_trades: int = 5) -> tuple[int, float]:
    """Pick the grid index with the best TRAIN Sharpe. Reads train window only.

    Returns (chosen_index, train_sharpe). Prefers params that actually trade in the
    train window; falls back to best Sharpe regardless; ties break to the lower
    index (the simpler/earlier config) for determinism.
    """
    scored = []          # (sharpe, -idx, idx, traded_enough)
    for idx in sorted(results):
        res = results[idx]
        if res is None:
            continue
        m = window_metrics(res, train_start, train_end)
        if m is None:
            continue
        traded = m["round_trips"] >= min_train_trades
        scored.append((m["sharpe"], traded, idx))
    if not scored:
        return 0, float("nan")
    traded_set = [s for s in scored if s[1]]
    pool = traded_set if traded_set else scored
    # max sharpe, tie -> lower idx
    best = max(pool, key=lambda s: (s[0], -s[2]))
    return best[2], best[0]


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    chosen: list[tuple]            # per fold: (idx, params, train_sharpe)
    oos_returns: pd.Series         # stitched OOS daily returns (chronological)
    oos_metrics: dict              # metrics on the stitched OOS curve
    oos_round_trips: int           # total OOS round trips across folds
    oos_total_return: float
    bench_oos_total_return: float
    oos_exposure: float


def walk_forward(results: dict, grid: list[dict], folds=None) -> WalkForwardResult:
    folds = folds or vcfg.FOLDS
    chosen = []
    oos_ret_parts = []
    total_rt = 0
    strat_growth = 1.0
    bench_growth = 1.0

    for (tr_s, tr_e, te_s, te_e) in folds:
        idx, train_sr = select_on_train(results, grid, tr_s, tr_e)
        params = grid[idx] if idx < len(grid) else {}
        chosen.append((idx, params, train_sr))

        res = results.get(idx)
        if res is None:
            continue
        # OOS scoring of the selected param on the unseen test window
        oos_m = window_metrics(res, te_s, te_e)
        oos_r = window_returns(res, te_s, te_e)
        if len(oos_r) > 1:
            oos_ret_parts.append(oos_r)
        if oos_m is not None:
            total_rt += int(oos_m["round_trips"])
            strat_growth *= (1.0 + oos_m["total_return"])
            bench_growth *= (1.0 + oos_m["benchmark_total_return"])

    if oos_ret_parts:
        stitched = pd.concat(oos_ret_parts)
        stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()
    else:
        stitched = pd.Series(dtype=float)

    oos_metrics = _series_metrics(stitched)
    exposure = float((stitched != 0).mean()) if len(stitched) else 0.0

    return WalkForwardResult(
        chosen=chosen,
        oos_returns=stitched,
        oos_metrics=oos_metrics,
        oos_round_trips=total_rt,
        oos_total_return=strat_growth - 1.0,
        bench_oos_total_return=bench_growth - 1.0,
        oos_exposure=exposure,
    )


def _series_metrics(returns: pd.Series) -> dict:
    """Sharpe/Sortino/total-return/maxDD on a bare return series (per-period)."""
    if returns is None or len(returns) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "total_return": 0.0,
                "max_drawdown": 0.0, "ann_return": 0.0, "n_obs": 0}
    ppy = vcfg.PERIODS_PER_YEAR
    mean, sd = returns.mean(), returns.std()
    sharpe = float(np.sqrt(ppy) * mean / sd) if sd > 0 else 0.0
    downside = returns[returns < 0].std()
    sortino = float(np.sqrt(ppy) * mean / downside) if downside and downside > 0 else 0.0
    curve = (1.0 + returns).cumprod()
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    total = float(curve.iloc[-1] - 1.0)
    years = len(returns) / ppy
    ann = float(curve.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "total_return": total,
            "max_drawdown": dd, "ann_return": ann, "n_obs": int(len(returns))}
