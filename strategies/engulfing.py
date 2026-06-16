"""
Bullish Engulfing candlestick reversal.

Pattern : after a down move, a small bearish candle is followed by a larger bullish
          candle whose body completely engulfs the prior body — buyers seizing control.
Entry   : long when, in a prior downtrend, today's body engulfs yesterday's bearish body.
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (5), trend_window (5).
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (engulfing pattern).
Backward-looking: reads bars i and i-1 plus a trailing SMA; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class Engulfing(Strategy):
    def __init__(self, hold: int = 5, trend_window: int = 5):
        self.hold, self.trend_window = hold, trend_window
        self.name = "Bullish Engulfing"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, c = df["open"], df["close"]
        prev_bear = c.shift(1) < o.shift(1)
        cur_bull = c > o
        engulf = (c >= o.shift(1)) & (o <= c.shift(1))
        pattern = prev_bear & cur_bull & engulf & cd.trend_down(c, self.trend_window)
        df["pattern"] = pattern
        df["cs_pos"] = cd.hold_position(pattern, self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
