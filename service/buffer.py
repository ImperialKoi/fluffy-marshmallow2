"""
In-memory rolling bar buffer.

The always-on service runs on LIVE streamed bars only — it never loads the backtest
CSV. We keep just the last N minute bars per symbol in a bounded deque so memory stays
tiny (target: a 1 GB box). The stream-consumer thread writes; the fast/slow loops read,
so access is guarded by a lock.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd


class BarBuffer:
    def __init__(self, maxlen: int = 240):
        self.maxlen = int(maxlen)
        self._data: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.maxlen))
        self._lock = threading.Lock()
        self._last_update: dict[str, datetime] = {}

    def add_bar(self, symbol: str, ts, o, h, l, c, v) -> None:
        sym = symbol.upper()
        # Normalize EVERY timestamp to a stdlib datetime in UTC so the buffer is
        # homogeneous. Warm-up bars arrive as pandas Timestamps and live websocket
        # bars as stdlib datetimes; mixing the two broke pd.DatetimeIndex (pandas 2.x
        # refuses a mixed tz-aware sequence unless utc=True).
        t = pd.Timestamp(ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        ts = t.to_pydatetime()
        with self._lock:
            self._data[sym].append((ts, float(o), float(h), float(l), float(c), float(v)))
            self._last_update[sym] = datetime.now(timezone.utc)

    def frame(self, symbol: str):
        """A backward-looking OHLCV DataFrame for the symbol, or None if empty."""
        with self._lock:
            rows = list(self._data.get(symbol.upper(), ()))
        if not rows:
            return None
        idx = [r[0] for r in rows]
        return pd.DataFrame(
            {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
             "low": [r[3] for r in rows], "close": [r[4] for r in rows],
             "volume": [r[5] for r in rows]},
            # utc=True coerces any residual tz-aware/naive mix to a single UTC index
            index=pd.to_datetime(idx, utc=True),
        )

    def latest_price(self, symbol: str):
        with self._lock:
            d = self._data.get(symbol.upper())
            return d[-1][4] if d else None

    def n(self, symbol: str) -> int:
        with self._lock:
            return len(self._data.get(symbol.upper(), ()))

    def symbols(self) -> list[str]:
        with self._lock:
            return [s for s, d in self._data.items() if d]

    def last_update(self, symbol: str):
        with self._lock:
            return self._last_update.get(symbol.upper())

    def seconds_since_any_update(self):
        with self._lock:
            if not self._last_update:
                return None
            newest = max(self._last_update.values())
        return (datetime.now(timezone.utc) - newest).total_seconds()
