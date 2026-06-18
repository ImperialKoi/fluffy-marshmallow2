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
                  universe, mode: str, min_bars: int = 30, exit_settings=None) -> dict:
    """One fast-loop tick. Returns a summary dict (also suitable for logging/tests)."""
    result = {"halted": False, "signals": {}, "protective": [], "flat": [], "exits": []}

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

    # 2. DETERMINISTIC EXITS (no LLM): sell any managed long that breaches its risk
    #    frame (stop-loss / take-profit / support floor / resistance ceiling / crash).
    exited = set()
    if exit_settings is not None and getattr(exit_settings, "enabled", True):
        result["exits"] = _run_exits(buffer, broker, inventory, mode, exit_settings)
        exited = {e["symbol"] for e in result["exits"] if e.get("done")}

    # 3. deterministic detection over the buffer (reused strategy; detection only)
    for sym in universe:
        try:
            df = buffer.frame(sym)            # inside the try: a bad frame for one
            if df is None or len(df) < min_bars:  # symbol must not abort the whole tick
                continue                          # (protective reconcile must still run)
            prepared = fast_strategy.prepare(df.copy())
            result["signals"][sym] = int(fast_strategy.signal(prepared, len(prepared) - 1))
        except Exception as e:  # noqa: BLE001 — one symbol's detection must not break the tick
            log.debug("fast detect failed for %s: %s", sym, e)

    # 4. protective resting-order reconciliation (skip names we just exited)
    result["protective"] = protective.reconcile(broker, inventory, mode, skip=exited)
    return result


def _run_exits(buffer, broker, inventory, mode, settings) -> list[dict]:
    """Evaluate every managed long against its deterministic risk frame; sell on breach."""
    from alpaca.trading.enums import OrderSide
    from service.risk_exits import evaluate_exit
    out = []
    try:
        positions = broker.list_positions()
    except Exception as e:  # noqa: BLE001
        log.warning("exit engine: broker read failed: %s", e)
        return out
    open_orders = None
    for p in positions:
        sym = p["symbol"].upper()
        qty = int(float(p.get("qty", 0)))
        if qty <= 0:
            continue
        if inventory is not None and not inventory.is_managed(sym):
            continue
        avg = float(p.get("avg_entry_price") or 0.0)
        last = buffer.latest_price(sym)
        if last is None:
            last = float(p.get("current_price") or 0.0)
        if not (avg > 0 and last > 0):
            continue
        decision = evaluate_exit(avg, last, buffer.frame(sym), settings)
        if not decision["exit"]:
            continue
        rec = {"symbol": sym, "qty": qty, "reason": decision["reason"],
               "last": decision["last"], "levels": decision["levels"], "done": False}
        if mode == "dry":
            rec["action"] = "would_sell"
            log.info("[EXIT dry] would SELL %d %s (%s) last=%s levels=%s",
                     qty, sym, decision["reason"], decision["last"], decision["levels"])
        else:
            if open_orders is None:
                try:
                    open_orders = broker.get_open_orders()
                except Exception:  # noqa: BLE001
                    open_orders = []
            for o in open_orders:               # cancel resting protective orders first
                if o.get("symbol", "").upper() == sym:
                    broker.cancel_order(o["id"])
            try:
                broker.submit_market_order(sym, qty, OrderSide.SELL)
                rec["action"], rec["done"] = "sold", True
                if inventory is not None:
                    try:
                        inventory.meta.set(sym, expected_qty=0)
                    except Exception:  # noqa: BLE001
                        pass
                log.warning("[EXIT] SOLD %d %s (%s) last=%s levels=%s", qty, sym,
                            decision["reason"], decision["last"], decision["levels"])
            except Exception as e:  # noqa: BLE001
                rec["action"], rec["error"] = "sell_failed", str(e)
                log.warning("[EXIT] sell failed for %s: %s", sym, e)
        out.append(rec)
    return out
