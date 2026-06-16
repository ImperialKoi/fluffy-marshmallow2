"""
Concrete strategies.

  * SMACrossover      - trend following: long when fast SMA is above slow SMA.
  * RSIMeanReversion  - contrarian: buy oversold, sell when it recovers.
  * BuyAndHold        - the benchmark you must beat to justify any of this.

These are deliberately simple and well known. The point of the demo is a *correct
engine*, not a magic strategy. Real edges come later from the AI decision layer,
and only ever after honest out-of-sample validation.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class SMACrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow = fast, slow
        self.name = f"SMA({fast}/{slow}) crossover"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_fast"] = ind.sma(df["close"], self.fast)
        df["sma_slow"] = ind.sma(df["close"], self.slow)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        row = df.iloc[i]
        if pd.isna(row["sma_slow"]):       # not enough history yet
            return 0
        return 1 if row["sma_fast"] > row["sma_slow"] else 0


class RSIMeanReversion(Strategy):
    """Buy when RSI is oversold; exit once it climbs back above the exit level."""

    def __init__(self, window: int = 14, oversold: float = 30.0, exit_level: float = 55.0):
        self.window, self.oversold, self.exit_level = window, oversold, exit_level
        self.name = f"RSI({window}) mean-reversion <{oversold:.0f}/>{exit_level:.0f}"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = ind.rsi(df["close"], self.window)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df["rsi"].iloc[i]
        if pd.isna(r):
            return 0
        if r < self.oversold:
            return 1                       # oversold -> go long
        if r > self.exit_level:
            return 0                       # recovered -> step aside
        # In between: hold whatever we already decided last bar.
        prev = df["rsi"].iloc[i - 1] if i > 0 else r
        return 1 if prev < self.oversold else 0


class BuyAndHold(Strategy):
    """Benchmark: buy on day one, never sell."""

    name = "Buy & Hold"

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return 1
