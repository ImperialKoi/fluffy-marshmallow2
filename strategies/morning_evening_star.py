"""
Morning Star candlestick reversal (three-bar).

Pattern : a bearish candle, then a small-bodied "star" that gaps/pulls lower, then a
          strong bullish candle closing well into the first body — a classic bottom.
          (The bearish mirror is the Evening Star; this strategy trades the bullish side.)
Entry   : long when the three-bar Morning Star completes in a prior downtrend.
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (5), star_frac (0.5 = star body <= half the first body), trend_window (5).
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (morning/evening star).
Backward-looking: reads bars i, i-1, i-2 plus a trailing SMA; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class MorningStar(Strategy):
    def __init__(self, hold: int = 5, star_frac: float = 0.5, trend_window: int = 5):
        self.hold, self.star_frac, self.trend_window = hold, star_frac, trend_window
        self.name = "Morning Star"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, c = df["open"], df["close"]
        b = cd.body(o, c)
        first_bear = c.shift(2) < o.shift(2)
        small_star = b.shift(1) <= self.star_frac * b.shift(2)
        third_bull = c > o
        midpoint_first = (o.shift(2) + c.shift(2)) / 2.0
        closes_in = c > midpoint_first
        pattern = (first_bear & small_star & third_bull & closes_in
                   & cd.trend_down(c, self.trend_window))
        df["pattern"] = pattern
        df["cs_pos"] = cd.hold_position(pattern, self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
