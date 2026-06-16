"""
Piercing Line candlestick reversal.

Pattern : after a downtrend, a bearish candle is followed by a bullish candle that
          opens below the prior low but closes back above the midpoint of the prior
          body — buyers "piercing" the decline. (The bearish mirror is Dark Cloud Cover.)
Entry   : long when the Piercing Line completes in a prior downtrend.
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (5), trend_window (5).
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (piercing / dark cloud).
Backward-looking: reads bars i and i-1 plus a trailing SMA; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class PiercingLine(Strategy):
    def __init__(self, hold: int = 5, trend_window: int = 5):
        self.hold, self.trend_window = hold, trend_window
        self.name = "Piercing Line"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, c = df["open"], df["close"]
        prev_bear = c.shift(1) < o.shift(1)
        cur_bull = c > o
        open_below = o < c.shift(1)
        mid_prev = (o.shift(1) + c.shift(1)) / 2.0
        close_above_mid = (c > mid_prev) & (c < o.shift(1))
        pattern = (prev_bear & cur_bull & open_below & close_above_mid
                   & cd.trend_down(c, self.trend_window))
        df["pattern"] = pattern
        df["cs_pos"] = cd.hold_position(pattern, self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
