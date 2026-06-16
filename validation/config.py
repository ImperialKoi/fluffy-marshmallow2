"""
Validation configuration: chronological splits, walk-forward folds, per-strategy
parameter grids, cost scenarios, and the mechanical graduation thresholds.

These are the *only* knobs for the validation phase. The split/fold dates and the
graduation thresholds are deliberately fixed up front so the test set is never
tuned (see VALIDATION.md, "the cardinal rule").
"""

from __future__ import annotations

from dataclasses import dataclass

import config as proj  # project-level config (costs, sizing, stops)


# --------------------------------------------------------------------------- #
# Chronological split (overfitting visibility): train then a later, unseen test
# --------------------------------------------------------------------------- #
SPLIT_TRAIN = ("2013-01-01", "2016-12-31")
SPLIT_TEST = ("2017-01-01", "2018-12-31")

# --------------------------------------------------------------------------- #
# Walk-forward folds: rolling 2y train -> next 1y test, stepped 1y.
# Selection/tuning happens on each fold's TRAIN window only; the TEST window is
# scored once and never influences a train-period choice.
# (date_train_start, date_train_end, date_test_start, date_test_end)
# --------------------------------------------------------------------------- #
FOLDS = [
    ("2013-02-01", "2015-01-31", "2015-02-01", "2016-01-31"),
    ("2014-02-01", "2016-01-31", "2016-02-01", "2017-01-31"),
    ("2015-02-01", "2017-01-31", "2017-02-01", "2018-02-28"),
]


# --------------------------------------------------------------------------- #
# Cost scenarios. "normal" = the project's standing assumptions; "stress" cranks
# friction up to see which edges survive realistic costs.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostScenario:
    name: str
    commission_bps: float
    slippage_bps: float


COSTS = {
    "normal": CostScenario("normal", proj.COMMISSION_BPS, proj.SLIPPAGE_BPS),
    "stress": CostScenario("stress", 5.0, 25.0),
}


# --------------------------------------------------------------------------- #
# Graduation thresholds (applied mechanically; do NOT loosen to manufacture wins)
# --------------------------------------------------------------------------- #
MIN_ROUND_TRIPS = 30          # OOS round trips below this -> "insufficient evidence"
MIN_SYMBOLS_EVALUATED = 10    # need enough symbols with adequate trades to judge
MAJORITY = 0.50               # must beat baselines on > this fraction of symbols
ROBUST_PLATEAU_FRAC = 0.50    # >= this fraction of grid within ROBUST_TOL of best train SR
ROBUST_TOL = 0.25            # "near best" = within 25% of the best train Sharpe
DSR_THRESHOLD = 0.95          # Deflated Sharpe Ratio must exceed this probability
BONFERRONI_ALPHA = 0.05       # family-wise alpha before the #strategies correction


# --------------------------------------------------------------------------- #
# Per-strategy parameter grids (small, sensible — swept on TRAIN data only).
# Names match strategies/registry.py. Strategies absent here run with their
# defaults (single config; robustness reported as N/A — no tunable spike risk).
# Keep each grid small to bound runtime; the point is plateau-vs-spike, not search.
# --------------------------------------------------------------------------- #
GRIDS: dict[str, list[dict]] = {
    "sma": [{"fast": 10, "slow": 30}, {"fast": 20, "slow": 50},
            {"fast": 20, "slow": 100}, {"fast": 50, "slow": 150}],
    "ema": [{"fast": 10, "slow": 30}, {"fast": 12, "slow": 26},
            {"fast": 20, "slow": 50}, {"fast": 20, "slow": 100}],
    "triple_ma": [{"fast": 5, "medium": 15, "slow": 40},
                  {"fast": 10, "medium": 20, "slow": 50},
                  {"fast": 20, "medium": 50, "slow": 100}],
    "macd": [{"fast": 12, "slow": 26, "signal": 9},
             {"fast": 8, "slow": 21, "signal": 5},
             {"fast": 19, "slow": 39, "signal": 9}],
    "rsi": [{"window": 7, "oversold": 20}, {"window": 14, "oversold": 30},
            {"window": 14, "oversold": 25}, {"window": 21, "oversold": 35}],
    "connors_rsi": [{"oversold": 5}, {"oversold": 10}, {"oversold": 15}],
    "bollinger_reversion": [{"window": 10, "num_std": 1.5}, {"window": 20, "num_std": 2.0},
                            {"window": 20, "num_std": 2.5}, {"window": 30, "num_std": 2.0}],
    "bollinger_breakout": [{"window": 10, "num_std": 1.5}, {"window": 20, "num_std": 2.0},
                           {"window": 20, "num_std": 2.5}, {"window": 30, "num_std": 2.0}],
    "zscore": [{"window": 10, "entry_z": 1.5}, {"window": 20, "entry_z": 2.0},
               {"window": 20, "entry_z": 2.5}, {"window": 40, "entry_z": 2.0}],
    "donchian": [{"entry": 10, "exit_window": 5}, {"entry": 20, "exit_window": 10},
                 {"entry": 55, "exit_window": 20}],
    "keltner": [{"ema_window": 20, "atr_window": 10, "mult": 1.5},
                {"ema_window": 20, "atr_window": 10, "mult": 2.0},
                {"ema_window": 20, "atr_window": 10, "mult": 2.5}],
    "supertrend": [{"window": 7, "mult": 3.0}, {"window": 10, "mult": 3.0},
                   {"window": 10, "mult": 2.0}, {"window": 14, "mult": 2.0}],
    "chandelier": [{"window": 22, "mult": 2.0}, {"window": 22, "mult": 3.0},
                   {"window": 14, "mult": 3.0}],
    "stochastic": [{"k_window": 14, "oversold": 20}, {"k_window": 21, "oversold": 20},
                   {"k_window": 14, "oversold": 30}],
    "williams_r": [{"window": 10}, {"window": 14}, {"window": 21}],
    "cci": [{"window": 14}, {"window": 20}, {"window": 30}],
    "roc": [{"window": 20}, {"window": 60}, {"window": 90}, {"window": 120}],
    "mfi": [{"window": 10}, {"window": 14}, {"window": 21}],
    "vwap_reversion": [{"window": 10, "band": 0.02}, {"window": 20, "band": 0.02},
                       {"window": 20, "band": 0.03}],
    "high_52w": [{"window": 200, "give_back": 0.10}, {"window": 252, "give_back": 0.10},
                 {"window": 252, "give_back": 0.15}],
    "fibonacci": [{"swing_window": 40}, {"swing_window": 60}, {"swing_window": 90}],
    "down_days": [{"n_down": 2}, {"n_down": 3}, {"n_down": 4}],
    "sr_breakout": [{"p_threshold": 0.50}, {"p_threshold": 0.55}, {"p_threshold": 0.60}],
    "sr_reversion": [{"p_threshold": 0.35}, {"p_threshold": 0.40}, {"p_threshold": 0.45}],
}


def grid_for(name: str) -> list[dict]:
    """Param grid for a strategy, or [default] if none defined."""
    g = GRIDS.get(name)
    return g if g else [{}]


# Risk settings reused verbatim from the project so validation matches deployment.
RISK = dict(
    position_fraction=proj.POSITION_FRACTION,
    stop_loss_pct=proj.STOP_LOSS_PCT,
    take_profit_pct=proj.TAKE_PROFIT_PCT,
    max_drawdown_kill=proj.MAX_DRAWDOWN_KILL,
)
INITIAL_CASH = proj.INITIAL_CASH
PERIODS_PER_YEAR = proj.TRADING_DAYS_PER_YEAR
RISK_FREE = proj.RISK_FREE_RATE
RANDOM_BASELINE_SEEDS = (1, 2, 3)   # median over a few seeds for a stable baseline
