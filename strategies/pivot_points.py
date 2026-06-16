"""
Floor-trader pivot points.

Pattern : classic pivots derive today's reference levels from the PRIOR period's
          high/low/close: pivot P = (H+L+C)/3, with R1/S1 around it. Trading above
          the pivot is a bullish bias; a push through R1 is intraday strength.
          On daily bars the "prior period" is the prior day.
Entry   : long when close > the pivot P computed from the prior bar.
Exit    : flat when close < S1 (prior-bar support).
Params  : (none — standard floor pivots).
Source  : floor-trader (classic) pivot points.
Backward-looking: levels use the PRIOR bar's HLC via shift(1); position is ffilled.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy


class PivotPoints(Strategy):
    def __init__(self):
        self.name = "Pivot points (floor)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ph, pl, pc = df["high"].shift(1), df["low"].shift(1), df["close"].shift(1)
        pivot = (ph + pl + pc) / 3.0
        r1 = 2 * pivot - pl
        s1 = 2 * pivot - ph
        df["pivot"], df["pivot_r1"], df["pivot_s1"] = pivot, r1, s1
        entry = df["close"] > pivot
        exit_ = df["close"] < s1
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["pp_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["pp_pos"].iloc[i])
