"""
Bullish Harami candlestick reversal.

Pattern : a large bearish candle followed by a small bullish candle whose body is
          contained inside the prior body ("inside" day) — momentum stalling, a turn hint.
Entry   : long when, in a prior downtrend, a small bullish body sits inside the prior bearish body.
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (5), trend_window (5).
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (harami).
Backward-looking: reads bars i and i-1 plus a trailing SMA; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class Harami(Strategy):
    def __init__(self, hold: int = 5, trend_window: int = 5):
        self.hold, self.trend_window = hold, trend_window
        self.name = "Bullish Harami"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, c = df["open"], df["close"]
        prev_bear = c.shift(1) < o.shift(1)
        cur_bull = c > o
        inside = (c <= o.shift(1)) & (o >= c.shift(1))
        pattern = prev_bear & cur_bull & inside & cd.trend_down(c, self.trend_window)
        df["pattern"] = pattern
        df["cs_pos"] = cd.hold_position(pattern, self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
