"""
VWAP reversion.

Pattern : the volume-weighted average price is a fair-value magnet. When price
          trades a meaningful distance below rolling VWAP it often reverts back to it.
Entry   : long when close is more than `band` below rolling VWAP.
Exit    : flat when close climbs back to/above VWAP.
Params  : window (20), band (0.02 = 2% below VWAP).
Source  : VWAP reversion (institutional execution benchmark used as fair value).
          NOTE: daily-bar rolling VWAP, not a true intraday session VWAP.
Backward-looking: rolling VWAP; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class VWAPReversion(Strategy):
    def __init__(self, window: int = 20, band: float = 0.02):
        self.window, self.band = window, band
        self.name = f"VWAP reversion({window}, {band*100:.0f}%)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        vwap = ind.vwap_rolling(df["high"], df["low"], df["close"], df["volume"], self.window)
        df["vwap"] = vwap
        entry = df["close"] < vwap * (1 - self.band)
        exit_ = df["close"] >= vwap
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["vwap_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["vwap_pos"].iloc[i])
