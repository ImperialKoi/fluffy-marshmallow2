"""
Inverse Head-and-Shoulders chart pattern.

Pattern : three troughs — a low (left shoulder), a deeper low (head), a higher low
          (right shoulder) — with two interim peaks forming the neckline. A break
          above the neckline confirms a bullish reversal. (The standard head &
          shoulders top is the bearish mirror; this trades the inverse/bullish form.)
Entry   : long when close breaks above the neckline of a valid inverse H&S.
Exit    : flat when close falls back below the right-shoulder low (invalidation).
Params  : left/right (4), tol (0.05, shoulder symmetry), lookback (160).
Source  : Thomas Bulkowski, "Encyclopedia of Chart Patterns" (head-and-shoulders).
Backward-looking: uses only swing pivots confirmed by bar i (see features/patterns).
"""

import pandas as pd

from strategies.base import Strategy
from features import patterns as pat


class InverseHeadShoulders(Strategy):
    def __init__(self, left: int = 4, right: int = 4, tol: float = 0.05, lookback: int = 160):
        self.left, self.right, self.tol, self.lookback = left, right, tol, lookback
        self.name = "Inverse Head & Shoulders"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        entry, exit_ = pat.inverse_head_shoulders(df, self.left, self.right,
                                                  self.tol, self.lookback)
        df["hs_pos"] = pat.to_position(entry, exit_)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["hs_pos"].iloc[i])
