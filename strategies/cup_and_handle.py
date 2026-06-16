"""
Cup-and-Handle chart pattern.

Pattern : a rounded "cup" (a U-shaped decline and recovery back to a prior high/rim)
          followed by a shallow "handle" pullback, then a breakout above the rim — a
          bullish continuation pattern popularised by William O'Neil.
Entry   : long when close breaks above the rim after a cup of valid depth.
Exit    : flat when close falls back below the cup's low (invalidation).
Params  : left/right (4), rim_tol (0.05), min_depth (0.12), max_depth (0.45), lookback (200).
Source  : William O'Neil, "How to Make Money in Stocks" (cup-with-handle).
          Heuristic detector — documented as approximate, not an exhaustive recogniser.
Backward-looking: uses only swing pivots confirmed by bar i (see features/patterns).
"""

import pandas as pd

from strategies.base import Strategy
from features import patterns as pat


class CupAndHandle(Strategy):
    def __init__(self, left: int = 4, right: int = 4, rim_tol: float = 0.05,
                 min_depth: float = 0.12, max_depth: float = 0.45, lookback: int = 200):
        self.left, self.right, self.rim_tol = left, right, rim_tol
        self.min_depth, self.max_depth, self.lookback = min_depth, max_depth, lookback
        self.name = "Cup and Handle"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        entry, exit_ = pat.cup_and_handle(df, self.left, self.right, self.rim_tol,
                                          self.min_depth, self.max_depth, self.lookback)
        df["cup_pos"] = pat.to_position(entry, exit_)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cup_pos"].iloc[i])
