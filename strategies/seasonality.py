"""
Seasonality / turn-of-month effect.

Pattern : equity returns have historically clustered around the turn of the month
          (the last calendar day or two plus the first few of the next month), a
          well-documented calendar anomaly. This strategy is long only in that window.
Entry   : long on the first `first_days` and last `last_days` CALENDAR days of a month.
Exit    : flat on all other days.
Params  : first_days (3), last_days (2). Set weekday=N instead to hold a single
          weekday (e.g. the Monday effect), 0=Mon .. 4=Fri.
Source  : turn-of-the-month effect (Ariel 1987; Lakonishok & Smidt 1988).
Backward-looking: depends ONLY on the bar's calendar date (day / days_in_month),
          never on any future bar — so it is trivially lookahead-free.
"""

import pandas as pd

from strategies.base import Strategy


class Seasonality(Strategy):
    def __init__(self, first_days: int = 3, last_days: int = 2, weekday: int = None):
        self.first_days, self.last_days, self.weekday = first_days, last_days, weekday
        self.name = (f"Seasonality(weekday={weekday})" if weekday is not None
                     else f"Turn-of-month(first {first_days}/last {last_days})")

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx = df.index
        if self.weekday is not None:
            df["season_long"] = (idx.weekday == self.weekday).astype(float)
            return df
        dom = idx.day                       # calendar day of month (1-based)
        dim = idx.days_in_month             # total calendar days in this month
        is_first = dom <= self.first_days
        is_last = dom > (dim - self.last_days)
        df["season_long"] = (is_first | is_last).astype(float)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["season_long"].iloc[i])
