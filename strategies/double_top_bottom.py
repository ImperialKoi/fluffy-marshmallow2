"""
Double Bottom chart pattern (the "W").

Pattern : price makes a low, bounces to an interim peak, falls to a second low near
          the first, then rallies. The break above the interim peak (the neckline)
          confirms a double-bottom reversal. (Double tops are the bearish mirror;
          this long-only strategy trades the bullish double bottom.)
Entry   : long when close breaks above the neckline between two similar swing lows.
Exit    : flat when close falls back below the pattern's lows (invalidation).
Params  : left/right (4, swing strength), tol (0.04, low-similarity), lookback (120).
Source  : Thomas Bulkowski, "Encyclopedia of Chart Patterns" (double bottom).
Backward-looking: uses only swing pivots confirmed by bar i (see features/patterns).
"""

import pandas as pd

from strategies.base import Strategy
from features import patterns as pat


class DoubleBottom(Strategy):
    def __init__(self, left: int = 4, right: int = 4, tol: float = 0.04, lookback: int = 120):
        self.left, self.right, self.tol, self.lookback = left, right, tol, lookback
        self.name = "Double Bottom"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        entry, exit_ = pat.double_bottom(df, self.left, self.right, self.tol, self.lookback)
        df["db_pos"] = pat.to_position(entry, exit_)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["db_pos"].iloc[i])
