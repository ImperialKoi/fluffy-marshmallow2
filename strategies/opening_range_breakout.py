"""
Opening-range breakout (daily adaptation).

Pattern : the classic intraday ORB buys when price breaks above the first N minutes'
          high. On DAILY bars there is no intraday range, so this is the honest
          daily analogue: treat the prior bar's range as the "opening range" and
          buy a breakout above the prior day's high.
Entry   : long when close > prior bar's high.
Exit    : flat when close < prior bar's low.
Params  : (none — uses the single prior bar). lookback>1 widens the reference range.
Source  : Opening-Range Breakout (Toby Crabel / Tony Crabel), adapted to daily bars.
          Documented adaptation: intraday ORB is not reproducible on daily data.
Backward-looking: compares to prior bar(s) via shift(1); position is forward-filled.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class OpeningRangeBreakout(Strategy):
    def __init__(self, lookback: int = 1):
        self.lookback = lookback
        self.name = f"Opening-range breakout(daily, {lookback}-bar range)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ref_high = ind.rolling_max(df["high"], self.lookback).shift(1)
        ref_low = ind.rolling_min(df["low"], self.lookback).shift(1)
        entry = df["close"] > ref_high
        exit_ = df["close"] < ref_low
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["orb_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["orb_pos"].iloc[i])
