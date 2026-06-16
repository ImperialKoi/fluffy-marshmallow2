"""
Support/Resistance reversion (fade levels unlikely to break).

Pattern : the mirror of the breakout strategy. When price sits on a support floor
          that the breakout-probability model (features/levels.py) judges UNLIKELY
          to break (few catalysts: low volume, no downside momentum), the high-odds
          play is a bounce off the floor back up toward the ceiling.
Entry   : long when close is within `near` of the nearest support AND that floor's
          P(break down) <= `p_threshold` (the level should hold).
Exit    : flat when close reaches within `near` of the nearest resistance (target),
          or closes below the support (the floor broke after all).
Params  : p_threshold (0.40), near (0.03), left/right (3), tol (0.02), buffer (0.01).
Source  : range / support-bounce trading + the breakout-probability heuristics in
          features/levels.py (low P(break) -> level respected -> fade it).
Backward-looking: levels & probabilities use only pivots confirmed by bar i; position
          is forward-filled.
Note    : the prepared frame exposes `p_break_support` / `p_break_resistance` columns
          for analysis of the probability behind each decision.
"""

import numpy as np
import pandas as pd

from strategies.base import Strategy
from features import levels as lv


class SRReversion(Strategy):
    def __init__(self, p_threshold: float = 0.40, near: float = 0.03, left: int = 3,
                 right: int = 3, tol: float = 0.02, buffer: float = 0.01):
        self.p_threshold, self.near = p_threshold, near
        self.left, self.right, self.tol, self.buffer = left, right, tol, buffer
        self.name = f"S/R reversion (fade, P<={p_threshold:.2f})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        levels = lv.compute_levels(df, left=self.left, right=self.right, tol=self.tol)
        df = pd.concat([df, levels], axis=1)
        near_support = df["dist_support"] <= self.near
        respected = df["p_break_support"] <= self.p_threshold
        entry = near_support & respected
        at_resistance = df["dist_resistance"] <= self.near
        broke = df["close"] < df["support"] * (1 - self.buffer)
        exit_ = at_resistance | broke
        raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=df.index)
        df["srr_pos"] = raw.ffill().fillna(0.0)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["srr_pos"].iloc[i])
