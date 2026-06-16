"""
Volume-weighted momentum.

Pattern : price momentum is more trustworthy when it is backed by above-average
          volume. Combine a positive rate-of-change with a volume expansion filter
          so we only ride moves that volume confirms.
Entry   : long when ROC(window) > 0 AND volume > vol_mult * average volume.
Exit    : flat when ROC <= 0.
Params  : window (20), vol_mult (1.2), vol_window (20).
Source  : volume-confirmed momentum (a standard "price + volume" trend filter).
Backward-looking: ROC and rolling volume average; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class VolumeWeightedMomentum(Strategy):
    def __init__(self, window: int = 20, vol_mult: float = 1.2, vol_window: int = 20):
        self.window, self.vol_mult, self.vol_window = window, vol_mult, vol_window
        self.name = f"Volume-weighted momentum({window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        r = ind.roc(df["close"], self.window)
        vol_ma = df["volume"].rolling(self.vol_window, min_periods=1).mean()
        df["roc"] = r
        entry = (r > 0) & (df["volume"] > self.vol_mult * vol_ma)
        exit_ = r <= 0
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["vwm_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["vwm_pos"].iloc[i])
