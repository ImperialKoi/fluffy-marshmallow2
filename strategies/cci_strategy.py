"""
CCI (Commodity Channel Index) strategy.

Pattern : CCI measures how far the typical price sits from its moving average in
          units of mean deviation. +/-100 are the standard extremes.
Entry   : long when CCI crosses up through -100 (emerging from oversold).
Exit    : flat when CCI crosses down through +100 (overbought rollover).
Params  : window (20).
Source  : Donald Lambert's Commodity Channel Index.
Backward-looking: CCI from rolling typical-price stats; position is forward-filled.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class CCIStrategy(Strategy):
    def __init__(self, window: int = 20, lower: float = -100.0, upper: float = 100.0):
        self.window, self.lower, self.upper = window, lower, upper
        self.name = f"CCI({window}) reversion"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = ind.cci(df["high"], df["low"], df["close"], self.window)
        df["cci"] = c
        entry = (c.shift(1) <= self.lower) & (c > self.lower)
        exit_ = (c.shift(1) >= self.upper) & (c < self.upper)
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["cci_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cci_pos"].iloc[i])
