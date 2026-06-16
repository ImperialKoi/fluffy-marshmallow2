"""
Williams %R (momentum mean-reversion).

Pattern : %R is an inverted stochastic on a -100..0 scale; below -80 is oversold,
          above -20 overbought.
Entry   : long when %R crosses back up above -80 (leaving oversold).
Exit    : flat when %R rises above -20 (overbought).
Params  : window (14), oversold (-80), overbought (-20).
Source  : Larry Williams' %R.
Backward-looking: %R from rolling high/low; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class WilliamsR(Strategy):
    def __init__(self, window: int = 14, oversold: float = -80.0, overbought: float = -20.0):
        self.window, self.oversold, self.overbought = window, oversold, overbought
        self.name = f"Williams %R({window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        wr = ind.williams_r(df["high"], df["low"], df["close"], self.window)
        df["willr"] = wr
        entry = (wr.shift(1) <= self.oversold) & (wr > self.oversold)
        exit_ = wr > self.overbought
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["willr_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["willr_pos"].iloc[i])
