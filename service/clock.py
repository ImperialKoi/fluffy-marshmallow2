"""
Market-hours gating.

The service only streams/scans/trades during market hours. `MarketClock` uses Alpaca's
authoritative clock for REGULAR hours; with `extended_hours=True` it also opens during
the pre/post-market windows (a NY-time heuristic; the websocket simply won't deliver
bars on a true holiday, and orders would be rejected, so this is a safe loosening).

Tests inject a fake clock object exposing `is_open()`, so this class needs no network
under test; `in_extended_window` is a pure function that is unit-tested directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timezone

log = logging.getLogger("service.clock")

try:
    from zoneinfo import ZoneInfo
    _NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _NY = None

# regular session 09:30–16:00 ET; pre 04:00–09:30; post 16:00–20:00
_PRE_OPEN = time(4, 0)
_REG_OPEN = time(9, 30)
_REG_CLOSE = time(16, 0)
_POST_CLOSE = time(20, 0)


def in_extended_window(now_utc: datetime) -> bool:
    """True if `now_utc` is within a weekday pre- or post-market window (NY time).
    (Does not account for holidays — see the module note.)"""
    if _NY is None:
        return False
    ny = now_utc.astimezone(_NY)
    if ny.weekday() >= 5:        # Sat/Sun
        return False
    t = ny.time()
    return (_PRE_OPEN <= t < _REG_OPEN) or (_REG_CLOSE <= t < _POST_CLOSE)


class MarketClock:
    def __init__(self, broker, extended_hours: bool = False):
        self.broker = broker
        self.extended_hours = extended_hours

    def is_open(self) -> bool:
        try:
            info = self.broker.get_clock()
        except Exception as e:  # noqa: BLE001 — if the clock call fails, fail CLOSED (safe)
            log.warning("clock check failed (%s) -> treating market as CLOSED", e)
            return False
        if info.get("is_open"):
            return True
        if self.extended_hours and in_extended_window(datetime.now(timezone.utc)):
            return True
        return False

    def status(self) -> dict:
        try:
            info = self.broker.get_clock()
        except Exception:  # noqa: BLE001
            return {"is_open": False, "error": "clock unavailable"}
        return {"is_open": self.is_open(), "regular_open": bool(info.get("is_open")),
                "next_open": str(info.get("next_open")), "next_close": str(info.get("next_close")),
                "extended_hours": self.extended_hours}
