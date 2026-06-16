"""
Protective resting-order manager — the real "act at any moment".

Whenever a managed long position exists, we keep a server-side protective SELL order
resting at Alpaca (GTC) so it fires at the exchange even if the bot or the box is down.
By default that's a stop-loss at the RiskManager's stop distance below entry; optionally
a trailing stop, a take-profit limit, or an OCO (take-profit + stop) bracket.

`reconcile()` is idempotent and the heart of the backstop:
  * for each MANAGED long position, check for an existing protective sell order;
  * if one already covers the position (qty matches) -> do nothing (NO double-submit);
  * if missing -> place the configured protective order;
  * if stale (qty mismatch after a resize) -> cancel it and re-place.
In dry mode it computes and logs the intended actions but places nothing.

Only MANAGED positions are protected (the inventory wall-off); unmanaged/discovered
positions are left untouched, consistent with the rest of the system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config

log = logging.getLogger("service.protective")

_PROTECTIVE_TYPES = ("stop", "trailing_stop", "limit", "stop_limit")


@dataclass
class ProtectiveSettings:
    enabled: bool = True
    stop_pct: float = 0.08
    trailing_pct: float = None       # fraction, e.g. 0.05; None = off
    take_profit_pct: float = None    # fraction; None = off
    bracket_oco: bool = False        # OCO (take-profit limit + stop) if a TP is set

    @classmethod
    def from_config(cls) -> "ProtectiveSettings":
        return cls(
            enabled=getattr(config, "PROTECT_ENABLED", True),
            stop_pct=getattr(config, "PROTECT_STOP_PCT", config.STOP_LOSS_PCT),
            trailing_pct=getattr(config, "PROTECT_TRAILING_PCT", None),
            take_profit_pct=getattr(config, "PROTECT_TAKE_PROFIT_PCT", None),
            bracket_oco=getattr(config, "PROTECT_BRACKET_OCO", False),
        )

    def kind(self) -> str:
        if self.bracket_oco and self.take_profit_pct:
            return "oco"
        if self.trailing_pct:
            return "trailing"
        return "stop"


class ProtectiveOrderManager:
    def __init__(self, settings: ProtectiveSettings = None):
        self.s = settings or ProtectiveSettings.from_config()

    def stop_price(self, avg_price: float) -> float:
        return round(avg_price * (1.0 - self.s.stop_pct), 2)

    def take_profit_price(self, avg_price: float) -> float:
        return round(avg_price * (1.0 + (self.s.take_profit_pct or 0.0)), 2)

    @staticmethod
    def _protective_sells(open_orders, symbol):
        sym = symbol.upper()
        out = []
        for o in open_orders:
            if o.get("symbol", "").upper() != sym:
                continue
            if not str(o.get("side", "")).upper().endswith("SELL"):
                continue
            otype = str(o.get("type", "")).lower()
            if any(t in otype for t in _PROTECTIVE_TYPES) or o.get("order_class", "").lower() == "oco":
                out.append(o)
        return out

    def reconcile(self, broker, inventory=None, mode: str = "paper") -> list[dict]:
        """Ensure every managed long position has a resting protective order."""
        if not self.s.enabled:
            return []
        try:
            positions = broker.list_positions()
            open_orders = broker.get_open_orders()
        except Exception as e:  # noqa: BLE001 — never let a broker hiccup crash the loop
            log.warning("protective reconcile: broker read failed: %s", e)
            return []

        actions = []
        for p in positions:
            sym = p["symbol"].upper()
            qty = int(float(p.get("qty", 0)))
            if qty <= 0:                      # long-only protection
                continue
            if inventory is not None and not inventory.is_managed(sym):
                continue                      # wall-off: don't touch unmanaged positions
            avg = float(p.get("avg_entry_price") or p.get("current_price") or 0.0)
            existing = self._protective_sells(open_orders, sym)
            covered = [o for o in existing if int(float(o.get("qty", 0))) == qty]
            if covered:
                actions.append({"symbol": sym, "action": "ok", "qty": qty})
                continue

            stale = [o for o in existing if int(float(o.get("qty", 0))) != qty]
            stop = self.stop_price(avg)
            intent = {"symbol": sym, "qty": qty, "kind": self.s.kind(),
                      "stop_price": stop, "stale_to_cancel": [o["id"] for o in stale]}

            if mode == "dry":
                intent["action"] = "would_place"
                actions.append(intent)
                continue

            for o in stale:                   # clear stale before re-placing (no dup)
                broker.cancel_order(o["id"])
            try:
                self._place(broker, sym, qty, avg)
                intent["action"] = "placed"
            except Exception as e:  # noqa: BLE001
                intent["action"] = "place_failed"
                intent["error"] = str(e)
                log.warning("failed to place protective order for %s: %s", sym, e)
            actions.append(intent)
        return actions

    def _place(self, broker, symbol, qty, avg):
        kind = self.s.kind()
        if kind == "oco":
            broker.submit_oco_exit(symbol, qty, self.take_profit_price(avg), self.stop_price(avg))
        elif kind == "trailing":
            broker.submit_trailing_stop(symbol, qty, self.s.trailing_pct * 100.0)
        else:
            broker.submit_stop_order(symbol, qty, self.stop_price(avg))
