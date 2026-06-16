"""
Three White Soldiers candlestick continuation/reversal.

Pattern : three consecutive strong bullish candles, each opening within the prior
          body and closing near its high at progressively higher levels — sustained
          buying. (The bearish mirror is Three Black Crows.)
Entry   : long when three white soldiers complete.
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (5), body_frac (0.6 = body >= 60% of range for "strong").
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (three white soldiers).
Backward-looking: reads bars i, i-1, i-2; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class ThreeWhiteSoldiers(Strategy):
    def __init__(self, hold: int = 5, body_frac: float = 0.6):
        self.hold, self.body_frac = hold, body_frac
        self.name = "Three White Soldiers"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        rng = cd.candle_range(h, l)
        strong_bull = (c > o) & (cd.body(o, c) >= self.body_frac * rng)
        s0, s1, s2 = strong_bull, strong_bull.shift(1), strong_bull.shift(2)
        higher = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        opens_within = (o < c.shift(1)) & (o > o.shift(1))
        pattern = s0 & s1 & s2 & higher & opens_within
        df["pattern"] = pattern.fillna(False)
        df["cs_pos"] = cd.hold_position(df["pattern"], self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
