"""
Bullish Marubozu candlestick.

Pattern : a marubozu is a candle with (almost) no shadows — open ~= low and close ~=
          high for a bullish marubozu, i.e. buyers controlled the whole session. It
          signals strong momentum continuation.
Entry   : long when a bullish marubozu prints (tiny shadows, large body).
Exit    : after `hold` bars (and the shared risk stop-loss).
Params  : hold (3), shadow_frac (0.05 = each shadow <= 5% of range).
Source  : Steve Nison, "Japanese Candlestick Charting Techniques" (marubozu).
Backward-looking: reads bar i geometry only; long for `hold` bars after.
"""

import pandas as pd

from strategies.base import Strategy
from features import candles as cd


class Marubozu(Strategy):
    def __init__(self, hold: int = 3, shadow_frac: float = 0.05):
        self.hold, self.shadow_frac = hold, shadow_frac
        self.name = "Bullish Marubozu"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        rng = cd.candle_range(h, l)
        bull = c > o
        tiny_up = cd.upper_shadow(o, h, c) <= self.shadow_frac * rng
        tiny_lo = cd.lower_shadow(o, l, c) <= self.shadow_frac * rng
        pattern = (bull & tiny_up & tiny_lo).fillna(False)
        df["pattern"] = pattern
        df["cs_pos"] = cd.hold_position(pattern, self.hold)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cs_pos"].iloc[i])
