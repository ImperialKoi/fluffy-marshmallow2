"""
Rate-of-Change momentum (absolute momentum / time-series momentum).

Pattern : if price is higher than it was N bars ago, momentum is positive and
          tends to persist (the well-documented time-series momentum effect).
Entry   : long when ROC(window) > threshold.
Exit    : flat when ROC <= threshold.
Params  : window (90), threshold (0.0).
Source  : rate-of-change / time-series ("absolute") momentum (Moskowitz, Ooi,
          Pedersen, 2012).
Backward-looking: ROC = close/close.shift(window); signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class ROCMomentum(Strategy):
    def __init__(self, window: int = 90, threshold: float = 0.0):
        self.window, self.threshold = window, threshold
        self.name = f"ROC({window}) momentum"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["roc"] = ind.roc(df["close"], self.window)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df["roc"].iloc[i]
        if pd.isna(r):
            return 0
        return 1 if r > self.threshold else 0
