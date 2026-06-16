"""
Donchian channel breakout (the classic Turtle system).

Pattern : channel = highest high / lowest low of the last N bars. A close above the
          prior N-bar high is a trend breakout; a close below the prior exit-window
          low closes the trade.
Entry   : long when close breaks above the prior `entry`-bar high.
Exit    : flat when close breaks below the prior `exit_window`-bar low.
Params  : entry (20), exit_window (10), allow_short (False).
Source  : Richard Donchian channels; the 20/10 rule is the Turtle Traders' system.
Backward-looking: channel uses rolling max/min shifted by 1 (excludes the current
          bar so we compare against history only); position is forward-filled state.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import indicators as ind


class DonchianBreakout(Strategy):
    def __init__(self, entry: int = 20, exit_window: int = 10, allow_short: bool = False):
        self.entry, self.exit_window, self.allow_short = entry, exit_window, allow_short
        self.name = f"Donchian breakout({entry}/{exit_window})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        upper, _, _ = ind.donchian(df["high"], df["low"], self.entry)
        _, lower, _ = ind.donchian(df["high"], df["low"], self.exit_window)
        # Compare to the channel formed by *prior* bars (shift by 1).
        up_prev = upper.shift(1)
        lo_prev = lower.shift(1)
        long_entry = df["close"] > up_prev
        long_exit = df["close"] < lo_prev
        raw = pd.Series(np.where(long_entry, 1.0, np.where(long_exit, 0.0, np.nan)),
                        index=df.index)
        df["dc_pos"] = raw.ffill().fillna(0.0)
        if self.allow_short:
            # symmetric short side using the same windows
            short_entry = df["close"] < lo_prev
            short_exit = df["close"] > up_prev
            sraw = pd.Series(np.where(short_entry, -1.0, np.where(short_exit, 0.0, np.nan)),
                             index=df.index).ffill().fillna(0.0)
            # net long/flat takes priority when both fire; otherwise use short
            df["dc_pos"] = np.where(df["dc_pos"] > 0, 1.0, sraw)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["dc_pos"].iloc[i])
