"""
Deterministic exit engine — runs every fast tick, uses NO LLM.

The LLM decides what to BUY (from news); exits must not depend on it. This module
gives every managed long position a hard, always-on risk frame and sells the moment a
level is breached, regardless of whether any LLM is reachable:

  * stop_loss      — floor on loss: last <= entry * (1 - stop_pct)
  * take_profit    — ceiling on gain: last >= entry * (1 + take_profit_pct)
  * support_break  — last breaks below the nearest SUPPORT (floor) from price structure
  * ceiling_reached— last reaches the nearest RESISTANCE (ceiling) while in profit
  * crash          — last falls crash_pct from a recent high (trailing crash protection)

Support/resistance reuse the project's backward-looking S/R module (features/levels.py)
computed on the live bar buffer. Everything here is pure given (avg_price, last, bars),
so it's unit-tested directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config
from features import levels as lv

log = logging.getLogger("service.risk_exits")

_MIN_SR_BARS = 20   # need some history before support/resistance is meaningful


@dataclass
class ExitSettings:
    enabled: bool = True
    stop_pct: float = 0.08
    take_profit_pct: float = 0.20
    crash_pct: float = 0.08
    crash_lookback: int = 30
    use_sr: bool = True
    support_break_buffer: float = 0.0
    ceiling_buffer: float = 0.0

    @classmethod
    def from_config(cls) -> "ExitSettings":
        return cls(
            enabled=getattr(config, "SERVICE_DETERMINISTIC_EXITS", True),
            stop_pct=getattr(config, "RISK_STOP_PCT", getattr(config, "STOP_LOSS_PCT", 0.08)),
            take_profit_pct=getattr(config, "RISK_TAKE_PROFIT_PCT", 0.20),
            crash_pct=getattr(config, "RISK_CRASH_PCT", 0.08),
            crash_lookback=getattr(config, "RISK_CRASH_LOOKBACK", 30),
            use_sr=getattr(config, "RISK_USE_SR", True),
            support_break_buffer=getattr(config, "RISK_SUPPORT_BREAK_BUFFER", 0.0),
            ceiling_buffer=getattr(config, "RISK_CEILING_BUFFER", 0.0),
        )


def support_resistance(df, settings: ExitSettings):
    """Nearest (support, resistance) from the buffer, or (None, None) if unavailable."""
    if not settings.use_sr or df is None or len(df) < _MIN_SR_BARS:
        return None, None
    try:
        out = lv.compute_levels(df)
        row = out.iloc[-1]
        sup = float(row["support"]) if row["support"] == row["support"] else None
        res = float(row["resistance"]) if row["resistance"] == row["resistance"] else None
        return sup, res
    except Exception as e:  # noqa: BLE001
        log.debug("S/R compute failed: %s", e)
        return None, None


def evaluate_exit(avg_price: float, last_price: float, df, settings: ExitSettings) -> dict:
    """Decide whether to exit a long. Returns {exit, reason, levels, last, avg}."""
    stop = avg_price * (1 - settings.stop_pct)
    take = avg_price * (1 + settings.take_profit_pct)
    floor, ceiling = support_resistance(df, settings)
    recent_high = None
    if df is not None and len(df) >= 2:
        recent_high = float(df["high"].tail(settings.crash_lookback).max())

    reason = None
    if last_price <= stop:
        reason = "stop_loss"                                   # floor (loss)
    elif last_price >= take:
        reason = "take_profit"                                 # ceiling (gain target)
    elif floor is not None and last_price < floor * (1 - settings.support_break_buffer):
        reason = "support_break"                               # crashed through the floor
    elif (ceiling is not None and last_price > avg_price
          and last_price >= ceiling * (1 - settings.ceiling_buffer)):
        reason = "ceiling_reached"                             # hit structural ceiling in profit
    elif (recent_high and recent_high > 0
          and last_price <= recent_high * (1 - settings.crash_pct)):
        reason = "crash"                                       # rapid drop from a recent high

    levels = {"stop": round(stop, 2), "take_profit": round(take, 2),
              "floor": round(floor, 2) if floor else None,
              "ceiling": round(ceiling, 2) if ceiling else None,
              "recent_high": round(recent_high, 2) if recent_high else None}
    return {"exit": reason is not None, "reason": reason, "levels": levels,
            "last": round(last_price, 2), "avg": round(avg_price, 2)}
