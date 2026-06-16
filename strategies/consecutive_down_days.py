"""
Consecutive down-days mean-reversion.

Pattern : after several down closes in a row, short-term selling tends to be
          exhausted and price bounces — a simple, robust mean-reversion edge that
          Larry Connors documented extensively (e.g. the "%b" / pullback systems).
Entry   : long after `n_down` consecutive lower closes (optionally above a trend MA).
Exit    : flat after the first up close, or after `max_hold` bars.
Params  : n_down (3), max_hold (5), trend_ma (200), use_trend_filter (True).
Source  : Larry Connors' short-term pullback / consecutive-down-days research.
Backward-looking: streak counts past closes only; position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class ConsecutiveDownDays(Strategy):
    def __init__(self, n_down: int = 3, max_hold: int = 5, trend_ma: int = 200,
                 use_trend_filter: bool = True):
        self.n_down, self.max_hold = n_down, max_hold
        self.trend_ma, self.use_trend_filter = trend_ma, use_trend_filter
        self.name = f"{n_down} down days reversion"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        s = ind.streak(df["close"])
        df["streak"] = s
        trend = ind.sma(df["close"], self.trend_ma)
        entry_cond = s <= -self.n_down
        if self.use_trend_filter:
            entry_cond = entry_cond & (df["close"] > trend)
        up_close = df["close"] > df["close"].shift(1)
        # position: enter long on the trigger, exit on first up close or after max_hold
        entry = entry_cond
        exit_ = up_close
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        pos = raw.ffill().fillna(0.0)
        # cap holding period
        pos = _cap_hold(pos.to_numpy(), self.max_hold)
        df["cdd_pos"] = pd.Series(pos, index=df.index)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["cdd_pos"].iloc[i])


def _cap_hold(pos, max_hold):
    """Force flat once a long position has been held `max_hold` bars (backward-looking)."""
    out = pos.copy()
    held = 0
    for i in range(len(out)):
        if out[i] > 0:
            held += 1
            if held > max_hold:
                out[i] = 0.0
        else:
            held = 0
    return out
