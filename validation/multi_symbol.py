"""
Multi-symbol runner.

Evaluates a strategy across many symbols and aggregates by CONSISTENCY, not by best
single result — a real edge shows up broadly, noise shows up on one name. For each
(strategy, symbol) it produces, under a given cost scenario:

  * a chronological train/test split (in-sample vs out-of-sample, shown separately
    so overfitting is visible),
  * a walk-forward OOS result (selection on each train window only; see splits.py),
  * a parameter-grid TRAIN surface (for robustness.py),
  * a matched-exposure random baseline and a buy-and-hold comparison.

One engine run per (strategy, symbol, param) serves the split AND every walk-forward
fold (we slice the full-history equity curve by window), which is what makes a
full-universe sweep tractable. Work is parallelized across symbols×strategies.
"""

from __future__ import annotations

import logging
import multiprocessing as mp

import numpy as np

from backtest.engine import run_backtest
from risk.manager import RiskManager
from strategies.registry import build

from . import config as vcfg
from . import data
from . import splits
from .baselines import RandomEntry

log = logging.getLogger("validation.multi_symbol")


def run_full(name, df, params, cost, symbol):
    """One full-history backtest with a FRESH RiskManager (it is stateful)."""
    strat = build(name, **params)
    risk = RiskManager(**vcfg.RISK)
    return run_backtest(df, strat, risk, vcfg.INITIAL_CASH,
                        cost.commission_bps, cost.slippage_bps, symbol)


def run_full_random(df, exposure, seed, cost, symbol):
    strat = RandomEntry(exposure=exposure, seed=seed)
    risk = RiskManager(**vcfg.RISK)
    return run_backtest(df, strat, risk, vcfg.INITIAL_CASH,
                        cost.commission_bps, cost.slippage_bps, symbol)


def evaluate_pair(name: str, symbol: str, cost_name: str = "normal") -> dict:
    """Full evaluation of one strategy on one symbol. Never raises (errors -> flags)."""
    cost = vcfg.COSTS[cost_name]
    grid = vcfg.grid_for(name)
    row = {"strategy": name, "symbol": symbol, "cost": cost_name, "error": ""}

    try:
        df = data.get_symbol(symbol)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"load:{e}"
        return row

    # one run per grid param (full history)
    results = {}
    for k, p in enumerate(grid):
        try:
            results[k] = run_full(name, df, p, cost, symbol)
        except Exception as e:  # noqa: BLE001
            results[k] = None
            row["error"] = f"run:{e}"

    # --- chronological split on the canonical (idx 0) params --------------- #
    base = results.get(0)
    if base is not None:
        is_m = splits.window_metrics(base, *vcfg.SPLIT_TRAIN)
        oos_m = splits.window_metrics(base, *vcfg.SPLIT_TEST)
        if is_m and oos_m:
            row.update(
                is_sharpe=is_m["sharpe"], is_return=is_m["total_return"],
                split_oos_sharpe=oos_m["sharpe"], split_oos_return=oos_m["total_return"],
                split_oos_rt=int(oos_m["round_trips"]),
                split_bh_return=oos_m["benchmark_total_return"],
                split_beats_bh=bool(oos_m["total_return"] > oos_m["benchmark_total_return"]),
                overfit_gap=is_m["sharpe"] - oos_m["sharpe"],
            )

    # --- walk-forward ----------------------------------------------------- #
    wf = splits.walk_forward(results, grid)
    row.update(
        wf_oos_sharpe=wf.oos_metrics["sharpe"],
        wf_oos_sortino=wf.oos_metrics["sortino"],
        wf_oos_return=wf.oos_total_return,
        wf_oos_maxdd=wf.oos_metrics["max_drawdown"],
        wf_oos_rt=int(wf.oos_round_trips),
        wf_exposure=wf.oos_exposure,
        wf_bench_return=wf.bench_oos_total_return,
        wf_beats_bh=bool(wf.oos_total_return > wf.bench_oos_total_return),
        wf_n_obs=wf.oos_metrics["n_obs"],
        wf_chosen=[(int(i), p, float(s) if s == s else None) for i, p, s in wf.chosen],
    )

    # --- robustness: TRAIN-window Sharpe across the grid ------------------ #
    surface = []
    for k, p in enumerate(grid):
        res = results.get(k)
        m = splits.window_metrics(res, *vcfg.SPLIT_TRAIN) if res is not None else None
        surface.append({"params": p,
                        "train_sharpe": (m["sharpe"] if m else float("nan")),
                        "train_rt": int(m["round_trips"]) if m else 0})
    row["grid_surface"] = surface

    # --- matched-exposure random baseline (walk-forward OOS) -------------- #
    rnd = []
    for seed in vcfg.RANDOM_BASELINE_SEEDS:
        try:
            rr = run_full_random(df, wf.oos_exposure, seed, cost, symbol)
            rwf = splits.walk_forward({0: rr}, [{}])
            rnd.append(rwf.oos_metrics["sharpe"])
        except Exception:  # noqa: BLE001
            continue
    row["wf_random_sharpe"] = float(np.median(rnd)) if rnd else float("nan")
    row["wf_beats_random"] = bool(
        row.get("wf_oos_sharpe", float("nan")) > row["wf_random_sharpe"]
    ) if rnd else False

    # stitched OOS returns (for pooled per-strategy Deflated Sharpe)
    row["oos_returns"] = {ts.isoformat(): float(v) for ts, v in wf.oos_returns.items()}
    return row


# --------------------------------------------------------------------------- #
# parallel universe sweep
# --------------------------------------------------------------------------- #
def _task(args):
    name, symbol, cost_name = args
    try:
        return evaluate_pair(name, symbol, cost_name)
    except Exception as e:  # noqa: BLE001 — never let one task kill the pool
        return {"strategy": name, "symbol": symbol, "cost": cost_name, "error": f"fatal:{e}"}


def run_universe(strategies, symbols, cost_name="normal", jobs=None, progress=True):
    tasks = [(s, sym, cost_name) for s in strategies for sym in symbols]
    jobs = jobs or max(1, mp.cpu_count() - 1)
    rows = []
    total = len(tasks)
    log.info("running %d (strategy x symbol) tasks on %d workers [cost=%s]",
             total, jobs, cost_name)
    if jobs == 1:
        for i, t in enumerate(tasks, 1):
            rows.append(_task(t))
            if progress and i % 50 == 0:
                print(f"  ... {i}/{total}", flush=True)
    else:
        with mp.Pool(jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=4), 1):
                rows.append(r)
                if progress and i % 50 == 0:
                    print(f"  ... {i}/{total}", flush=True)
    return rows
