"""
Gap-and-go (momentum continuation after a gap up).

Pattern : a stock that gaps up on strong volume and holds the gap often continues
          in the gap direction. We require an up-gap (today's open above yesterday's
          high) confirmed by the close holding above the open.
Entry   : long when open > prior high (gap up) AND close > open (gap held).
Exit    : flat when close < prior bar's low (gap failed / momentum lost).
Params  : min_gap (0.0 fractional gap size), vol_mult (1.0 = volume >= avg).
Source  : "gap and go" momentum trading (Ross Cameron / day-trading lore).
Backward-looking: gap measured vs prior bar via shift(1); ffilled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class GapAndGo(Strategy):
    def __init__(self, min_gap: float = 0.0, vol_mult: float = 1.0, vol_window: int = 20):
        self.min_gap, self.vol_mult, self.vol_window = min_gap, vol_mult, vol_window
        self.name = f"Gap-and-go(min_gap {min_gap*100:.1f}%)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prior_high = df["high"].shift(1)
        prior_low = df["low"].shift(1)
        vol_ma = df["volume"].rolling(self.vol_window, min_periods=1).mean()
        gap = df["open"] > prior_high * (1 + self.min_gap)
        held = df["close"] > df["open"]
        vol_ok = df["volume"] >= self.vol_mult * vol_ma
        entry = gap & held & vol_ok
        exit_ = df["close"] < prior_low
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["gap_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["gap_pos"].iloc[i])
