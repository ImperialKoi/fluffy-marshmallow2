"""
Fast loop (~60s): deterministic detection + protective backstop. NO LLM.

Each tick, using only the in-memory bar buffer (no CSV, no news):
  1. update the drawdown kill switch against real account equity; if tripped, GO FLAT
     (cancel protective orders + market-close managed positions) and stop — this is the
     responsive arm of the persisted kill switch;
  2. run a cheap deterministic strategy (reused from the registry, default Supertrend)
     over each symbol's buffered bars to flag bullish/bearish state — detection only,
     logged as a backstop signal, NOT the primary exit;
  3. reconcile protective resting orders so every managed long stays protected.

The real exits are the resting orders at the exchange; this loop makes sure they exist
and surfaces fast risk signals between hourly AI rebalances.
"""

from __future__ import annotations

import logging

log = logging.getLogger("service.fast_scan")


def go_flat(broker, inventory, mode: str) -> list[dict]:
    """Kill-switch backstop: cancel protective orders and market-close MANAGED longs.
    Idempotent — does nothing if already flat. In dry mode, logs intended actions only."""
    from alpaca.trading.enums import OrderSide
    actions = []
    try:
        positions = broker.list_positions()
        open_orders = broker.get_open_orders()
    except Exception as e:  # noqa: BLE001
        log.warning("go_flat: broker read failed: %s", e)
        return actions
    for p in positions:
        sym = p["symbol"].upper()
        qty = int(float(p.get("qty", 0)))
        if qty <= 0:
            continue
        if inventory is not None and not inventory.is_managed(sym):
            continue
        if mode == "dry":
            actions.append({"symbol": sym, "action": "would_flatten", "qty": qty})
            continue
        for o in open_orders:
            if o.get("symbol", "").upper() == sym:
                broker.cancel_order(o["id"])
        broker.submit_market_order(sym, qty, OrderSide.SELL)
        actions.append({"symbol": sym, "action": "flattened", "qty": qty})
    return actions


def run_fast_scan(*, buffer, broker, inventory, killswitch, protective, fast_strategy,
                  universe, mode: str, min_bars: int = 30) -> dict:
    """One fast-loop tick. Returns a summary dict (also suitable for logging/tests)."""
    result = {"halted": False, "signals": {}, "protective": [], "flat": []}

    # 1. kill switch against real equity
    try:
        equity = broker.equity()
        halted = killswitch.update(equity)
    except Exception as e:  # noqa: BLE001
        log.warning("fast_scan: equity/kill-switch check failed: %s", e)
        halted = killswitch.halted
    if halted:
        result["halted"] = True
        result["flat"] = go_flat(broker, inventory, mode)
        log.warning("KILL SWITCH active -> go flat (%d actions)", len(result["flat"]))
        return result

    # 2. deterministic detection over the buffer (reused strategy; detection only)
    for sym in universe:
        try:
            df = buffer.frame(sym)            # inside the try: a bad frame for one
            if df is None or len(df) < min_bars:  # symbol must not abort the whole tick
                continue                          # (step 3 protective reconcile must still run)
            prepared = fast_strategy.prepare(df.copy())
            sig = int(fast_strategy.signal(prepared, len(prepared) - 1))
            result["signals"][sym] = sig
        except Exception as e:  # noqa: BLE001 — one symbol's detection must not break the tick
            log.debug("fast detect failed for %s: %s", sym, e)

    # surface bearish flips on names we actually hold (backstop alert)
    if inventory is not None:
        for sym, sig in result["signals"].items():
            try:
                held = inventory.get(sym).get("qty", 0)
            except Exception:  # noqa: BLE001
                held = 0
            if held and sig < 1:
                log.info("fast scan: bearish/flat signal on held %s (sig=%d) — "
                         "protective stop is the exit", sym, sig)

    # 3. protective resting-order reconciliation
    result["protective"] = protective.reconcile(broker, inventory, mode)
    return result
