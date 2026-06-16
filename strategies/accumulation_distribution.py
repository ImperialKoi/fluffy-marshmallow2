"""
Accumulation/Distribution line trend.

Pattern : the A/D line weights each bar's volume by where the close finished within
          its range (close-location value), measuring accumulation vs distribution.
          A rising A/D line (above its MA) signals accumulation supporting price.
Entry   : long when the A/D line > its SMA.
Exit    : flat when the A/D line < its SMA.
Params  : ma_window (20).
Source  : Marc Chaikin's Accumulation/Distribution line.
Backward-looking: A/D is cumulative over past bars; SMA is rolling. Signal reads row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class AccumulationDistribution(Strategy):
    def __init__(self, ma_window: int = 20):
        self.ma_window = ma_window
        self.name = f"A/D line trend(SMA {ma_window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ad = ind.ad_line(df["high"], df["low"], df["close"], df["volume"])
        df["ad"] = ad
        df["ad_ma"] = ind.sma(ad, self.ma_window)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df.iloc[i]
        if pd.isna(r["ad_ma"]):
            return 0
        return 1 if r["ad"] > r["ad_ma"] else 0
