"""
ConnorsRSI mean-reversion.

Pattern : ConnorsRSI blends a 3-period price RSI, a 2-period RSI of the up/down
          streak, and the 100-day percent-rank of the 1-day ROC. Extreme low
          readings flag short-term oversold pullbacks inside an uptrend.
Entry   : long when ConnorsRSI < oversold (e.g. 10), ideally above a long MA filter.
Exit    : flat when ConnorsRSI > exit_level (e.g. 50) or a short RSI recovers.
Params  : oversold (10), exit_level (50), trend_ma (200), use_trend_filter (True).
Source  : Larry Connors / Connors Research; StockCharts ChartSchool ConnorsRSI.
Backward-looking: components use RSI/rolling percent-rank; signal reads i and i-1.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class ConnorsRSIStrategy(Strategy):
    def __init__(self, oversold: float = 10.0, exit_level: float = 50.0,
                 trend_ma: int = 200, use_trend_filter: bool = True):
        self.oversold, self.exit_level = oversold, exit_level
        self.trend_ma, self.use_trend_filter = trend_ma, use_trend_filter
        self.name = f"ConnorsRSI <{oversold:.0f}/>{exit_level:.0f}"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["crsi"] = ind.connors_rsi(df["close"])
        df["trend"] = ind.sma(df["close"], self.trend_ma)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df["crsi"].iloc[i]
        if pd.isna(r):
            return 0
        if self.use_trend_filter:
            trend = df["trend"].iloc[i]
            if pd.isna(trend) or df["close"].iloc[i] < trend:
                return 0
        if r < self.oversold:
            return 1
        if r > self.exit_level:
            return 0
        prev = df["crsi"].iloc[i - 1] if i > 0 else r
        return 1 if (prev < self.oversold and r <= self.exit_level) else 0
