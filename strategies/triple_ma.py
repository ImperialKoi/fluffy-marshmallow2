"""
Triple moving-average trend filter.

Pattern : three SMAs (fast/medium/slow). A clean bullish alignment
          fast > medium > slow signals a well-established uptrend, filtering out
          the whipsaws a single crossover suffers in choppy markets.
Entry   : long only when fast > medium > slow.
Exit    : flat as soon as the alignment breaks (fast <= medium).
Params  : fast (10), medium (20), slow (50).
Source  : triple moving average system (standard trend-following extension of the
          dual MA crossover).
Backward-looking: SMAs use rolling(min_periods); signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class TripleMA(Strategy):
    def __init__(self, fast: int = 10, medium: int = 20, slow: int = 50):
        if not (fast < medium < slow):
            raise ValueError("require fast < medium < slow")
        self.fast, self.medium, self.slow = fast, medium, slow
        self.name = f"Triple MA({fast}/{medium}/{slow})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma_fast"] = ind.sma(df["close"], self.fast)
        df["ma_med"] = ind.sma(df["close"], self.medium)
        df["ma_slow"] = ind.sma(df["close"], self.slow)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df.iloc[i]
        if pd.isna(r["ma_slow"]):
            return 0
        return 1 if (r["ma_fast"] > r["ma_med"] > r["ma_slow"]) else 0
