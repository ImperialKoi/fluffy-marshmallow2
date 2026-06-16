"""
MACD signal-line crossover (trend/momentum).

Pattern : MACD line = EMA(12) - EMA(26); signal line = EMA(9) of the MACD line.
          The histogram (MACD - signal) flipping positive marks momentum turning up.
Entry   : long when MACD line > signal line (histogram > 0).
Exit    : flat when MACD line <= signal line (allow_short=True flips to -1).
Params  : fast (12), slow (26), signal (9), allow_short (False).
Source  : Gerald Appel's Moving Average Convergence/Divergence.
Backward-looking: built from EMAs (ewm); signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class MACDCross(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 allow_short: bool = False):
        self.fast, self.slow, self.signal_w = fast, slow, signal
        self.allow_short = allow_short
        self.name = f"MACD({fast}/{slow}/{signal}) crossover"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        line, sig, hist = ind.macd(df["close"], self.fast, self.slow, self.signal_w)
        df["macd"], df["macd_signal"], df["macd_hist"] = line, sig, hist
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        h = df["macd_hist"].iloc[i]
        if pd.isna(h):
            return 0
        if h > 0:
            return 1
        return -1 if self.allow_short else 0
