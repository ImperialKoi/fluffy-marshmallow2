"""
Money Flow Index (volume-weighted RSI) mean-reversion.

Pattern : MFI is an RSI that incorporates volume via money flow. <20 is oversold,
          >80 overbought, with volume confirming the conviction behind the move.
Entry   : long when MFI crosses back up above the oversold line.
Exit    : flat when MFI rises above the overbought line.
Params  : window (14), oversold (20), overbought (80).
Source  : Gene Quong & Avrum Soudack's Money Flow Index.
Backward-looking: MFI from rolling typical-price * volume sums; position is ffilled.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class MoneyFlowIndex(Strategy):
    def __init__(self, window: int = 14, oversold: float = 20.0, overbought: float = 80.0):
        self.window, self.oversold, self.overbought = window, oversold, overbought
        self.name = f"MFI({window}) reversion"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        m = ind.mfi(df["high"], df["low"], df["close"], df["volume"], self.window)
        df["mfi"] = m
        entry = (m.shift(1) <= self.oversold) & (m > self.oversold)
        exit_ = m > self.overbought
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["mfi_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["mfi_pos"].iloc[i])
