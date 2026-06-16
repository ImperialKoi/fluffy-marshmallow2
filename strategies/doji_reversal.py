"""
Doji reversal candlestick.

Pattern : a doji has open ~= close (a tiny real body), signalling indecision. After a
          downtrend it can mark exhaustion and a turn back up.
Entry   : long when a doji prints in a prior downtrend.
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (3), body_frac (0.1 = body <= 10% of range), trend_window (5).
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (doji).
Backward-looking: reads bar i geometry plus a trailing SMA; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class DojiReversal(Strategy):
    def __init__(self, hold: int = 3, body_frac: float = 0.1, trend_window: int = 5):
        self.hold, self.body_frac, self.trend_window = hold, body_frac, trend_window
        self.name = "Doji reversal"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        is_doji = cd.body(o, c) <= self.body_frac * cd.candle_range(h, l)
        pattern = is_doji & cd.trend_down(c, self.trend_window)
        df["pattern"] = pattern
        df["cs_pos"] = cd.hold_position(pattern, self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
