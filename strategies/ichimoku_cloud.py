"""
Ichimoku Cloud (Kumo) trend strategy.

Pattern : the cloud (between Senkou Span A and B) is dynamic support/resistance.
          Price above the cloud = bullish regime; the Tenkan/Kijun cross times entries.
Entry   : long when close > top of the cloud AND conversion (Tenkan) > base (Kijun).
Exit    : flat when close falls back below the bottom of the cloud.
Params  : tenkan (9), kijun (26), senkou_b (52).
Source  : Goichi Hosoda's Ichimoku Kinko Hyo. The lagging (chikou) span is omitted
          on purpose because trading it requires future data.
Backward-looking: spans use shift(+kijun) (past data); signal at i reads only row i.
"""

import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class IchimokuCloud(Strategy):
    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan, self.kijun, self.senkou_b = tenkan, kijun, senkou_b
        self.name = f"Ichimoku({tenkan}/{kijun}/{senkou_b})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        conv, base, span_a, span_b = ind.ichimoku(
            df["high"], df["low"], df["close"], self.tenkan, self.kijun, self.senkou_b)
        df["tenkan"], df["kijun"] = conv, base
        df["cloud_top"] = pd.concat([span_a, span_b], axis=1).max(axis=1)
        df["cloud_bot"] = pd.concat([span_a, span_b], axis=1).min(axis=1)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        r = df.iloc[i]
        if pd.isna(r["cloud_top"]) or pd.isna(r["kijun"]):
            return 0
        if r["close"] > r["cloud_top"] and r["tenkan"] > r["kijun"]:
            return 1
        if r["close"] < r["cloud_bot"]:
            return 0
        # inside the cloud / mixed: stand aside
        return 0
