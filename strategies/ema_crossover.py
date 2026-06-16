"""
EMA crossover (trend following).

Pattern : two exponential moving averages; the faster one reacts quicker, so when
          it crosses above the slower one the short-term trend has turned up.
Entry   : long when fast EMA > slow EMA.
Exit    : flat when fast EMA <= slow EMA (set allow_short=True to flip to -1).
Params  : fast (12), slow (26), allow_short (False).
Source  : classic moving-average crossover (Appel-style EMA variant of the SMA
          crossover already in the repo). EMAs weight recent prices more heavily.
Backward-looking: EMAs use ewm(min_periods); signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class EMACrossover(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow, self.allow_short = fast, slow, allow_short
        self.name = f"EMA({fast}/{slow}) crossover"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = ind.ema(df["close"], self.fast)
        df["ema_slow"] = ind.ema(df["close"], self.slow)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        row = df.iloc[i]
        if pd.isna(row["ema_slow"]):
            return 0
        if row["ema_fast"] > row["ema_slow"]:
            return 1
        return -1 if self.allow_short else 0
