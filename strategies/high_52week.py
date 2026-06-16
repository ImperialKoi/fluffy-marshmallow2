"""
52-week high breakout (momentum).

Pattern : stocks making new 52-week highs tend to keep outperforming — the "52-week
          high momentum" anomaly. A close at/through the 1-year high signals strength.
Entry   : long when close >= the prior 252-bar high (a fresh 52-week high).
Exit    : flat when close falls more than `give_back` below the running high.
Params  : window (252), give_back (0.10 = 10% off the high).
Source  : George, "The 52-Week High and Momentum Investing" (2004).
Backward-looking: rolling 252-bar high (shifted to exclude current); ffilled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class High52Week(Strategy):
    def __init__(self, window: int = 252, give_back: float = 0.10):
        self.window, self.give_back = window, give_back
        self.name = f"52-week high breakout({window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prior_high = ind.rolling_max(df["high"], self.window).shift(1)
        run_high = ind.rolling_max(df["close"], self.window)
        df["high_52w"] = prior_high
        entry = df["close"] >= prior_high
        exit_ = df["close"] < run_high * (1 - self.give_back)
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["h52_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["h52_pos"].iloc[i])
