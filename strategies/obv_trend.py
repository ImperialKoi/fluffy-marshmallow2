"""
On-Balance Volume trend (volume confirmation).

Pattern : OBV accumulates volume on up days and subtracts it on down days. A rising
          OBV (above its own moving average) shows buying pressure backing the price.
Entry   : long when OBV > its SMA (volume trend up).
Exit    : flat when OBV < its SMA.
Params  : ma_window (20).
Source  : Joe Granville's On-Balance Volume.
Backward-looking: OBV is cumulative over past bars; SMA is rolling. Signal reads row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class OBVTrend(Strategy):
    def __init__(self, ma_window: int = 20):
        self.ma_window = ma_window
        self.name = f"OBV trend(SMA {ma_window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o = ind.obv(df["close"], df["volume"])
        df["obv"] = o
        df["obv_ma"] = ind.sma(o, self.ma_window)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df.iloc[i]
        if pd.isna(r["obv_ma"]):
            return 0
        return 1 if r["obv"] > r["obv_ma"] else 0
