"""
Ascending Triangle breakout chart pattern.

Pattern : a flat resistance line (a series of swing highs at a similar level) with
          rising swing lows beneath it forms an ascending triangle, a bullish
          continuation setup. The break above the flat top resolves it upward.
Entry   : long when close breaks above the flat resistance with rising lows beneath.
Exit    : flat when close falls back below the most recent rising low (invalidation).
Params  : left/right (4), tol (0.03, resistance flatness), lookback (120).
Source  : Thomas Bulkowski, "Encyclopedia of Chart Patterns" (ascending triangle).
Backward-looking: uses only swing pivots confirmed by bar i (see features/patterns).
"""

import pandas as pd

from strategies.base import Strategy
from features import patterns as pat


class TriangleBreakout(Strategy):
    def __init__(self, left: int = 4, right: int = 4, tol: float = 0.03, lookback: int = 120):
        self.left, self.right, self.tol, self.lookback = left, right, tol, lookback
        self.name = "Ascending Triangle breakout"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        entry, exit_ = pat.ascending_triangle(df, self.left, self.right,
                                              self.tol, self.lookback)
        df["tri_pos"] = pat.to_position(entry, exit_)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["tri_pos"].iloc[i])
