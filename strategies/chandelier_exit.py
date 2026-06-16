"""
Chandelier Exit (ATR volatility trailing-stop trend).

Pattern : a long trailing stop hung a multiple of ATR below the highest high since
          entry. While price stays above the chandelier line the uptrend is intact.
Entry   : long when close > the chandelier line (highest-high - mult*ATR).
Exit    : flat when close < the chandelier line.
Params  : window (22), mult (3.0).
Source  : Chuck LeBeau's Chandelier Exit (an ATR-based volatility stop).
Backward-looking: rolling highest-high and ATR; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class ChandelierExit(Strategy):
    def __init__(self, window: int = 22, mult: float = 3.0):
        self.window, self.mult = window, mult
        self.name = f"Chandelier Exit({window}, {mult}x ATR)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        a = ind.atr(df["high"], df["low"], df["close"], self.window)
        hh = ind.rolling_max(df["high"], self.window)
        line = hh - self.mult * a
        df["chandelier"] = line
        entry = df["close"] > line
        exit_ = df["close"] < line
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["ch_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["ch_pos"].iloc[i])
