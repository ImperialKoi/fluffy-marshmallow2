"""
Supertrend (ATR trend-following).

Pattern : an ATR-scaled band that flips above/below price. Its direction is the
          trend; the line itself doubles as a trailing stop.
Entry   : long when Supertrend direction is +1 (price above the line).
Exit    : flat when direction flips to -1 (allow_short=True flips to short).
Params  : window (10, ATR period), mult (3.0, ATR multiplier), allow_short (False).
Source  : Olivier Seban's Supertrend. Default 10/3 is the canonical setting.
Backward-looking: ATR + a recursion over bars <= i; signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class SupertrendStrategy(Strategy):
    def __init__(self, window: int = 10, mult: float = 3.0, allow_short: bool = False):
        self.window, self.mult, self.allow_short = window, mult, allow_short
        self.name = f"Supertrend({window}, {mult}x ATR)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        line, direction = ind.supertrend(df["high"], df["low"], df["close"],
                                          self.window, self.mult)
        df["supertrend"], df["st_dir"] = line, direction
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        d = df["st_dir"].iloc[i]
        if pd.isna(d):
            return 0
        if d > 0:
            return 1
        return -1 if self.allow_short else 0
