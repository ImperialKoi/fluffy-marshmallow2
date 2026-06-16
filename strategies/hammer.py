"""
Hammer candlestick reversal.

Pattern : after a decline, a candle with a small real body near the top and a long
          lower shadow (>= 2x the body) shows sellers were rejected — a bottoming hint.
Entry   : long when a hammer prints in a prior downtrend.
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (5), shadow_ratio (2.0), trend_window (5).
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (hammer / hanging man).
Backward-looking: reads bar i geometry plus a trailing SMA; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class Hammer(Strategy):
    def __init__(self, hold: int = 5, shadow_ratio: float = 2.0, trend_window: int = 5):
        self.hold, self.shadow_ratio, self.trend_window = hold, shadow_ratio, trend_window
        self.name = "Hammer"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        b = cd.body(o, c)
        lower = cd.lower_shadow(o, l, c)
        upper = cd.upper_shadow(o, h, c)
        small_body = b <= 0.4 * cd.candle_range(h, l)
        long_lower = lower >= self.shadow_ratio * b.replace(0.0, 1e-9)
        small_upper = upper <= b
        pattern = small_body & long_lower & small_upper & cd.trend_down(c, self.trend_window)
        df["pattern"] = pattern
        df["cs_pos"] = cd.hold_position(pattern, self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
