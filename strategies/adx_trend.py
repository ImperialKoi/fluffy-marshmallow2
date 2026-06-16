"""
ADX trend strategy (directional movement).

Pattern : ADX measures trend STRENGTH (not direction); +DI/-DI give direction.
          Only trade when a trend is actually present (ADX above a threshold).
Entry   : long when ADX > threshold AND +DI > -DI.
Exit    : flat when +DI <= -DI or ADX falls below the threshold.
Params  : window (14), adx_threshold (25).
Source  : Welles Wilder's Average Directional Index / Directional Movement System.
Backward-looking: ADX/DI use Wilder smoothing (ewm); signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class ADXTrend(Strategy):
    def __init__(self, window: int = 14, adx_threshold: float = 25.0):
        self.window, self.adx_threshold = window, adx_threshold
        self.name = f"ADX({window}) trend >{adx_threshold:.0f}"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        plus_di, minus_di, adx_ = ind.adx(df["high"], df["low"], df["close"], self.window)
        df["plus_di"], df["minus_di"], df["adx"] = plus_di, minus_di, adx_
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df.iloc[i]
        if pd.isna(r["adx"]):
            return 0
        if r["adx"] > self.adx_threshold and r["plus_di"] > r["minus_di"]:
            return 1
        return 0
