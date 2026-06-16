"""
Stochastic oscillator (momentum mean-reversion).

Pattern : %K compares the close to the recent high/low range; <20 is oversold,
          >80 overbought. Crossing up out of oversold times the entry.
Entry   : long when %K crosses above %D while %K is below the oversold line.
Exit    : flat when %K rises above the overbought line.
Params  : k_window (14), d_window (3), oversold (20), overbought (80).
Source  : George Lane's Stochastic Oscillator.
Backward-looking: %K/%D from rolling highs/lows; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class StochasticOscillator(Strategy):
    def __init__(self, k_window: int = 14, d_window: int = 3,
                 oversold: float = 20.0, overbought: float = 80.0):
        self.k_window, self.d_window = k_window, d_window
        self.oversold, self.overbought = oversold, overbought
        self.name = f"Stochastic({k_window}/{d_window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        k, d = ind.stochastic(df["high"], df["low"], df["close"],
                              self.k_window, self.d_window)
        df["stoch_k"], df["stoch_d"] = k, d
        cross_up = (k.shift(1) <= d.shift(1)) & (k > d) & (k < self.oversold)
        exit_ = k > self.overbought
        raw = pd.Series(np.where(cross_up, 1.0, np.where(exit_, 0.0, np.nan)),
                        index=df.index)
        df["stoch_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["stoch_pos"].iloc[i])
