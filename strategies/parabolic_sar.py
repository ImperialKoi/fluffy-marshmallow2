"""
Parabolic SAR (stop-and-reverse trend system).

Pattern : the SAR dots trail price and accelerate; price crossing the dots flips
          the trend. Designed for trend-following with a built-in trailing stop.
Entry   : long when close > SAR.
Exit    : flat when close <= SAR (allow_short=True flips to short, as Wilder intended).
Params  : af_step (0.02), af_max (0.20), allow_short (False).
Source  : Welles Wilder's Parabolic Stop And Reverse.
Backward-looking: SAR recursion uses only bars <= i; signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class ParabolicSAR(Strategy):
    def __init__(self, af_step: float = 0.02, af_max: float = 0.20,
                 allow_short: bool = False):
        self.af_step, self.af_max, self.allow_short = af_step, af_max, allow_short
        self.name = f"Parabolic SAR({af_step}/{af_max})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sar"] = ind.parabolic_sar(df["high"], df["low"], self.af_step, self.af_max)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df.iloc[i]
        if pd.isna(r["sar"]):
            return 0
        if r["close"] > r["sar"]:
            return 1
        return -1 if self.allow_short else 0
