"""
Bollinger Band mean-reversion.

Pattern : bands = SMA +/- k*std. In a range, a close at/under the lower band is a
          stretched move that tends to snap back toward the middle band.
Entry   : long when close < lower band.
Exit    : flat when close >= middle band (the SMA).
Params  : window (20), num_std (2.0).
Source  : John Bollinger's Bollinger Bands (mean-reversion / "band bounce" use).
Backward-looking: bands from rolling mean/std; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class BollingerReversion(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window, self.num_std = window, num_std
        self.name = f"Bollinger reversion({window}, {num_std}sd)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mid, up, lo = ind.bollinger(df["close"], self.window, self.num_std)
        df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, up, lo
        entry = df["close"] < lo
        exit_ = df["close"] >= mid
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["bb_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["bb_pos"].iloc[i])
