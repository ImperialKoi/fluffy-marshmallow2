"""
Support/Resistance breakout with breakout-probability gating.

Pattern : trade an actual break THROUGH a resistance ceiling, but only when the
          backward-looking breakout-probability model (features/levels.py) judges
          the break likely — i.e. the level has been tested enough, volume is
          expanding into it, momentum is pushing up, and price has coiled near it.
Entry   : long when close pushes above the prior bar's nearest resistance AND that
          level's P(break up) was >= `p_threshold`.
Exit    : flat when close falls back below the prior bar's nearest support.
Params  : p_threshold (0.55), left/right (3, swing strength), tol (0.02, clustering).
Source  : classic S&R breakout trading + the documented "more touches weaken a level
          / 50%+ volume expansion confirms a break" heuristics (see features/levels).
Backward-looking: every level and probability uses only pivots confirmed by bar i;
          entry/exit compare to PRIOR-bar levels (shift(1)); position is forward-filled.
Note    : the prepared frame exposes `p_break_resistance` / `p_break_support` columns
          so the probability behind each decision is available for analysis.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import levels as lv


class SRBreakout(Strategy):
    def __init__(self, p_threshold: float = 0.55, left: int = 3, right: int = 3,
                 tol: float = 0.02):
        self.p_threshold, self.left, self.right, self.tol = p_threshold, left, right, tol
        self.name = f"S/R breakout (P>={p_threshold:.2f})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        levels = lv.compute_levels(df, left=self.left, right=self.right, tol=self.tol)
        df = pd.concat([df, levels], axis=1)
        res_prev = df["resistance"].shift(1)
        sup_prev = df["support"].shift(1)
        p_prev = df["p_break_resistance"].shift(1)
        entry = (df["close"] > res_prev) & (p_prev >= self.p_threshold)
        exit_ = df["close"] < sup_prev
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["srb_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["srb_pos"].iloc[i])
