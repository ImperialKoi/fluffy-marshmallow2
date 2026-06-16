"""
Z-score mean-reversion.

Pattern : the z-score of price vs its rolling mean measures how many standard
          deviations price is stretched. Deep negative z-scores tend to revert up.
Entry   : long when z-score < -entry_z (stretched below the mean).
Exit    : flat when z-score >= exit_z (reverted back toward/above the mean).
Params  : window (20), entry_z (2.0), exit_z (0.0).
Source  : statistical mean-reversion / pairs-trading style z-score thresholds.
Backward-looking: z from rolling mean/std; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class ZScoreReversion(Strategy):
    def __init__(self, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.0):
        self.window, self.entry_z, self.exit_z = window, entry_z, exit_z
        self.name = f"Z-score reversion({window}, +/-{entry_z})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mean = ind.sma(df["close"], self.window)
        sd = df["close"].rolling(self.window, min_periods=self.window).std()
        z = (df["close"] - mean) / sd.replace(0.0, np.nan)
        df["zscore"] = z
        entry = z < -self.entry_z
        exit_ = z >= self.exit_z
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["z_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["z_pos"].iloc[i])
