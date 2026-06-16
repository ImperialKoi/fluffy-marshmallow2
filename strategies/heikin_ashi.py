"""
Heikin-Ashi trend strategy.

Pattern : Heikin-Ashi ("average bar") candles smooth price by averaging OHLC, making
          trends visually cleaner. A run of green HA candles (HA close > HA open)
          marks an uptrend; the first red candle warns it is ending.
Entry   : long when the HA candle is green and the prior HA candle was also green.
Exit    : flat on the first red HA candle (HA close < HA open).
Params  : (none — uses standard Heikin-Ashi construction).
Source  : Heikin-Ashi technique (Dan Valcu popularised it in the West).
Backward-looking: HA recursion uses bars <= i; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class HeikinAshi(Strategy):
    def __init__(self):
        self.name = "Heikin-Ashi trend"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ha_o, ha_h, ha_l, ha_c = ind.heikin_ashi(df["open"], df["high"], df["low"], df["close"])
        df["ha_open"], df["ha_close"] = ha_o, ha_c
        green = ha_c > ha_o
        entry = green & green.shift(1)
        exit_ = ~green
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["ha_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["ha_pos"].iloc[i])
