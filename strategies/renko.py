"""
Renko-style trend strategy.

Pattern : Renko charts ignore time and plot a brick only when price moves a fixed
          amount, filtering out noise. The brick direction is the trend. Here the
          brick size is ATR-based so it adapts to volatility.
Entry   : long when the current Renko brick direction is up (+1).
Exit    : flat when the brick direction turns down (-1).
Params  : atr_window (14), brick_mult (1.0, brick = brick_mult * ATR).
Source  : Renko charting (name from the Japanese "renga", brick).
Backward-looking: Renko recursion uses bars <= i; signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class Renko(Strategy):
    def __init__(self, atr_window: int = 14, brick_mult: float = 1.0):
        self.atr_window, self.brick_mult = atr_window, brick_mult
        self.name = f"Renko trend(ATR {atr_window} x {brick_mult})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        brick = self.brick_mult * ind.atr(df["high"], df["low"], df["close"], self.atr_window)
        # renko_trend treats warm-up NaN bricks as "no brick yet" (direction 0), so
        # we deliberately do NOT backfill — backfilling would leak future ATR values.
        df["renko_dir"] = ind.renko_trend(df["close"], brick)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return 1 if df["renko_dir"].iloc[i] > 0 else 0
