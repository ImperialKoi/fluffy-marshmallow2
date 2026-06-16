"""
Keltner Channel breakout (volatility trend).

Pattern : channel = EMA midline +/- multiple of ATR. A close above the upper band
          signals a volatility expansion to the upside (trend breakout).
Entry   : long when close > upper Keltner band.
Exit    : flat when close < EMA midline (give the trend room above the middle).
Params  : ema_window (20), atr_window (10), mult (2.0).
Source  : Chester Keltner's channel, ATR variant popularised by Linda Raschke.
Backward-looking: EMA + ATR; position is forward-filled state from bars <= i.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class KeltnerChannel(Strategy):
    def __init__(self, ema_window: int = 20, atr_window: int = 10, mult: float = 2.0):
        self.ema_window, self.atr_window, self.mult = ema_window, atr_window, mult
        self.name = f"Keltner breakout({ema_window}/{atr_window}, {mult}x)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mid, upper, lower = ind.keltner(df["high"], df["low"], df["close"],
                                        self.ema_window, self.atr_window, self.mult)
        df["kc_mid"], df["kc_up"], df["kc_lo"] = mid, upper, lower
        entry = df["close"] > upper
        exit_ = df["close"] < mid
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["kc_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["kc_pos"].iloc[i])
