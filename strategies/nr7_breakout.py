"""
NR7 breakout (narrowest range in 7 days).

Pattern : an NR7 bar has the smallest high-low range of the last 7 bars —
          volatility contraction that frequently precedes an expansion move. Trade
          the direction of the break of that NR7 bar.
Entry   : long when, after an NR7 bar, close breaks above that NR7 bar's high.
Exit    : flat when close breaks below the NR7 bar's low.
Params  : window (7).
Source  : Toby Crabel's NR7 (narrow-range) setup.
Backward-looking: range comparison and NR7 reference use shift(1); ffilled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class NR7Breakout(Strategy):
    def __init__(self, window: int = 7):
        self.window = window
        self.name = f"NR{window} breakout"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rng = df["high"] - df["low"]
        is_nr7 = rng == rng.rolling(self.window, min_periods=self.window).min()
        # reference levels come from the most recent NR7 bar (prior to current)
        nr_high = df["high"].where(is_nr7)
        nr_low = df["low"].where(is_nr7)
        ref_high = nr_high.ffill().shift(1)
        ref_low = nr_low.ffill().shift(1)
        entry = df["close"] > ref_high
        exit_ = df["close"] < ref_low
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["nr7_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["nr7_pos"].iloc[i])
