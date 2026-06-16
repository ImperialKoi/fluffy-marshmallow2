"""
Fibonacci retracement bounce.

Pattern : in an uptrend, pullbacks often find support at Fibonacci retracements
          (38.2% / 50% / 61.8%) of the prior swing. Buying the bounce inside that
          zone trades with the trend at better prices.
Entry   : long when, in an uptrend (close > long SMA), price has retraced into the
          38.2%-61.8% band of the recent swing low->high AND today closes up.
Exit    : flat when close falls below the swing low (retracement failed) or makes a
          new swing high (move complete) — captured by the rolling levels.
Params  : swing_window (60), trend_ma (100), fib_low (0.382), fib_high (0.618).
Source  : Fibonacci retracement (Leonardo Fibonacci ratios applied to swings).
Backward-looking: swing high/low from rolling windows; position is forward-filled.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class FibonacciRetracement(Strategy):
    def __init__(self, swing_window: int = 60, trend_ma: int = 100,
                 fib_low: float = 0.382, fib_high: float = 0.618):
        self.swing_window, self.trend_ma = swing_window, trend_ma
        self.fib_low, self.fib_high = fib_low, fib_high
        self.name = f"Fibonacci retracement({swing_window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        swing_hi = ind.rolling_max(df["high"], self.swing_window)
        swing_lo = ind.rolling_min(df["low"], self.swing_window)
        rng = (swing_hi - swing_lo).replace(0.0, np.nan)
        # retracement level (fraction of the move given back from the high)
        retr = (swing_hi - df["close"]) / rng
        trend = ind.sma(df["close"], self.trend_ma)
        uptrend = df["close"] > trend
        up_day = df["close"] > df["close"].shift(1)
        in_zone = (retr >= self.fib_low) & (retr <= self.fib_high)
        entry = uptrend & in_zone & up_day
        exit_ = (df["close"] >= swing_hi) | (df["close"] < swing_lo)
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["fib_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["fib_pos"].iloc[i])
