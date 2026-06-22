"""
Phase 3 forward-test harness: an AI, news-driven, cross-sectional paper strategy.

FORWARD-TEST ONLY — there is no historical backtest of the news signal. Each run:
  1. NewsPortfolioStrategy.evaluate(universe, as_of=now) -> per-symbol scores (LLM),
  2. deterministic constructor -> target weights (LLM never sizes trades),
  3. deterministic risk: drawdown kill switch + limits (always on),
  4. reconcile target weights -> target shares, diff vs Alpaca, submit market orders
     (looping the single-symbol reconciliation over the basket),
  5. log everything (scores, rationales, targets vs current, orders, equity) and
     track benchmarks (equal-weight universe + SPY) so evidence accumulates.

Safety:
  * --mode dry (compute+log, NO orders) | paper (default) | live (gated: --mode live
    AND ALPACA_ALLOW_LIVE=yes AND a typed confirmation).
  * Positions with no local record are NOT traded (managed=False wall-off); the AI
    only acts on flat universe names (which it then records) and managed positions.
  * On a NEW entry the live hype score is stamped as hype_at_entry (never backfilled);
    consumers treat a null hype_at_entry as unknown, not 0.

Commands:  (default) run a rebalance      |  report  -> CSV + chart vs benchmarks
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import config
from portfolio.constructor import construct_targets
from portfolio.risk import PortfolioLimits, KillSwitch

log = logging.getLogger("live_portfolio")


# --------------------------------------------------------------------------- #
# pure, testable helpers (no network)
# --------------------------------------------------------------------------- #
@dataclass
class Order:
    symbol: str
    target_shares: int
    current_shares: int
    delta: int
    side: str                # "BUY" / "SELL"
    target_weight: float
    new_entry: bool = False


def compute_orders(targets: dict, equity: float, prices: dict, current: dict,
                   managed: dict, universe) -> tuple[list[Order], list[tuple]]:
    """Reconcile target weights -> orders. Long-only.

    Wall-off rule: a held position that is NOT managed is never touched. Tradeable
    symbols are flat universe names (can open -> become managed) and managed names
    (adjust/close). `managed` maps symbol->bool; `current` maps symbol->shares.
    """
    uni = set(universe)
    relevant = uni | {s for s, q in current.items() if q}
    orders, skipped = [], []
    for sym in sorted(relevant):
        cur = int(current.get(sym, 0))
        held = cur != 0
        is_mgd = bool(managed.get(sym, False))
        if held and not is_mgd:
            skipped.append((sym, "unmanaged_position"))      # safety wall-off
            continue
        w = float(targets.get(sym, 0.0))
        px = prices.get(sym)
        if px is None or px <= 0:
            if w > 0 or held:
                skipped.append((sym, "no_price"))
            continue
        tgt = int((w * equity) // px) if w > 0 else 0
        delta = tgt - cur
        if delta != 0:
            orders.append(Order(sym, tgt, cur, delta,
                                "BUY" if delta > 0 else "SELL", w,
                                new_entry=(cur == 0 and tgt > 0)))
    return orders, skipped


def equal_weight_benchmark(start_equity: float, start_prices: dict, cur_prices: dict) -> float:
    """Equity of an equal-weight buy-and-hold of the universe since start."""
    rels = [cur_prices[s] / start_prices[s] for s in start_prices
            if s in cur_prices and start_prices[s] > 0 and cur_prices.get(s)]
    return start_equity * (sum(rels) / len(rels)) if rels else start_equity


def spy_benchmark(start_equity: float, spy0: float, spy: float) -> float:
    return start_equity * (spy / spy0) if spy0 and spy else start_equity


def portfolio_turnover(orders: list[Order], equity: float, prices: dict) -> float:
    if equity <= 0:
        return 0.0
    traded = sum(abs(o.delta) * prices.get(o.symbol, 0.0) for o in orders)
    return traded / equity


# --------------------------------------------------------------------------- #
# benchmark + logging state
# --------------------------------------------------------------------------- #
def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def init_or_load_benchmark(equity, prices, spy_price, path=None):
    """First run captures start equity + start prices; later runs reuse them."""
    path = path or config.AI_BENCH_STATE
    state = _load_json(path, None)
    if state is None:
        state = {"start_ts": datetime.now(timezone.utc).isoformat(),
                 "start_equity": equity,
                 "start_prices": {s: p for s, p in prices.items() if p},
                 "spy0": spy_price}
        _save_json(path, state)
    return state


def append_csv(path, row: dict, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- #
# one rebalance
# --------------------------------------------------------------------------- #
def run_once(broker, strategy, limits, killswitch, universe, mode, *,
             hype_tracker=None, inventory=None, as_of=None,
             tier_map=None, spec_limits=None):
    as_of = as_of or datetime.now(timezone.utc)
    equity = broker.equity()
    halted = killswitch.update(equity)
    held = _current_positions(broker, universe)

    # Evaluate FIRST so the strategy can expand the tradeable set (free-trade discovery
    # of new tickers; plus we always score current holdings so they can be sold/trimmed).
    degraded, signals, exposure, notes = False, {}, 0.0, []
    if halted:
        notes = ["KILL SWITCH: flat"]
        log.warning("kill switch ACTIVE (drawdown breach) -> flattening, no new entries")
    else:
        eval_universe = sorted(set(universe) | set(held))
        ps = strategy.evaluate(eval_universe, as_of=as_of)
        signals, exposure = ps.signals, ps.exposure_multiplier

    # Act on everything we scored OR hold (free-trade can add discovered tickers).
    act_universe = sorted(set(signals) | set(held) | set(universe))
    symbols = act_universe
    prices = _prices(broker, symbols)
    spy_price = _safe_price(broker, config.AI_BENCHMARK_SPY)
    managed = _managed_map(inventory, set(symbols))

    current_weights = {}
    if equity:
        for s in symbols:
            sh, px = int(held.get(s, 0)), prices.get(s)
            if sh and px and managed.get(s, False):
                current_weights[s] = sh * px / equity

    if halted:
        targets = {}                                   # flatten via compute_orders
    elif signals:
        ok_frac = sum(1 for s in signals.values() if s.ok) / len(signals)
        min_ok = getattr(config, "AI_MIN_OK_FRACTION", 0.5)
        if ok_frac < min_ok:
            # Degraded signal (e.g. an LLM/news outage): most symbols defaulted to
            # score 0. Do NOT rebalance on near-zero information — hold current
            # positions and wait for the next cycle.
            degraded = True
            targets = dict(current_weights)
            notes = [f"DEGRADED: only {ok_frac:.0%} of symbols returned a usable score "
                     f"(< {min_ok:.0%}) -> holding current positions, no rebalance"]
            log.warning("[DEGRADED] %.0f%% usable scores -> skipping rebalance, holding",
                        ok_frac * 100)
        else:
            cr = construct_targets(signals, limits, exposure_multiplier=exposure,
                                   current_weights=current_weights,
                                   turnover_cap=limits.turnover_cap,
                                   leverage=getattr(config, "RISK_MULTIPLIER", 1.0))
            targets, notes = cr.weights, cr.notes
            # Deterministic risk EXTENSION (final say): hard-cap the combined
            # speculative/penny sleeve and tighten per-name caps for speculative tiers.
            if tier_map:
                from portfolio.risk import enforce_speculative_sleeve
                targets, spec_notes = enforce_speculative_sleeve(
                    targets, tier_map, spec_limits)
                notes = notes + spec_notes
    else:
        targets = dict(current_weights)               # no signals -> hold

    if degraded:
        orders, skipped = [], []      # hold: place nothing this cycle
    else:
        orders, skipped = compute_orders(targets, equity, prices, held, managed, act_universe)

    # 4: execute (unless dry)
    placed = []
    if mode != "dry":
        placed = _execute(broker, orders, mode, hype_tracker, inventory, strategy)
    turnover = portfolio_turnover(orders, equity, prices)

    # 5: benchmarks + logging
    bench = init_or_load_benchmark(equity, prices, spy_price)
    ew = equal_weight_benchmark(bench["start_equity"], bench["start_prices"], prices)
    spy = spy_benchmark(bench["start_equity"], bench.get("spy0"), spy_price)
    _log_run(as_of, mode, equity, signals, targets, held, prices, orders,
             placed, ew, spy, turnover, exposure, halted)

    _print_summary(as_of, mode, equity, signals, targets, held, prices, orders,
                   skipped, ew, spy, exposure, halted, notes, strategy)
    return {"equity": equity, "orders": orders, "skipped": skipped,
            "targets": targets, "signals": signals, "ew": ew, "spy": spy,
            "halted": halted, "exposure": exposure}


def _execute(broker, orders, mode, hype_tracker, inventory, strategy):
    from alpaca.trading.enums import OrderSide
    placed = []
    # Resting protective SELL orders reserve the shares ("held_for_orders"), which blocks
    # the rebalance from selling them. Cancel a symbol's open orders before trading it to
    # free the shares; the slow loop re-runs protective.reconcile right after, re-placing
    # protection on the new quantity (brief unprotected window only during execution).
    trade_syms = {o.symbol.upper() for o in orders if o.delta != 0}
    if trade_syms:
        try:
            for oo in broker.get_open_orders():
                if oo.get("symbol", "").upper() in trade_syms:
                    broker.cancel_order(oo["id"])
        except Exception as e:  # noqa: BLE001
            log.warning("could not pre-cancel resting orders: %s", e)

    for o in orders:
        side = OrderSide.BUY if o.delta > 0 else OrderSide.SELL
        try:
            # per-order isolation: a single bad/non-tradable ticker (e.g. an AI-discovered
            # name Alpaca rejects) must not abort the rest of the basket.
            order = broker.submit_market_order(o.symbol, abs(o.delta), side)
        except Exception as e:  # noqa: BLE001
            log.warning("order rejected for %s (%s %d): %s", o.symbol, side, abs(o.delta), e)
            continue
        oid = str(getattr(order, "id", "")) if order else ""
        placed.append((o.symbol, o.delta, oid))
        if inventory is not None:
            _stamp_metadata(inventory, hype_tracker, o, strategy)
    return placed


def _stamp_metadata(inventory, hype_tracker, o: Order, strategy):
    """Record/refresh local metadata. On a NEW entry, stamp hype_at_entry with the
    live score at real entry time (never backfilled; left unknown if unavailable)."""
    try:
        fields = {"expected_qty": o.target_shares, "managed": True,
                  "strategy_tag": getattr(strategy, "name", "news_portfolio")}
        if o.new_entry:
            fields["entry_date"] = datetime.now(timezone.utc).date().isoformat()
            if hype_tracker is not None:
                try:
                    s = hype_tracker.score(o.symbol).get("score")
                    if s is not None and s == s:          # finite -> stamp; NaN -> leave null
                        fields["hype_at_entry"] = float(s)
                except Exception as e:  # noqa: BLE001
                    log.debug("hype_at_entry unavailable for %s: %s", o.symbol, e)
        inventory.meta.set(o.symbol, **fields)
    except Exception as e:  # noqa: BLE001
        log.warning("metadata stamp failed for %s: %s", o.symbol, e)


# --------------------------------------------------------------------------- #
# broker helpers
# --------------------------------------------------------------------------- #
def _current_positions(broker, universe):
    held = {}
    try:
        for p in broker.list_positions():
            held[p["symbol"].upper()] = int(p["qty"])
    except Exception:  # noqa: BLE001 — fall back to per-symbol for the universe
        for s in universe:
            try:
                held[s] = broker.position_shares(s)
            except Exception:  # noqa: BLE001
                held[s] = 0
    return held


def _managed_map(inventory, symbols):
    if inventory is None:
        return {s: True for s in symbols}   # no inventory -> treat all as tradeable
    return {s: inventory.is_managed(s) for s in symbols}


def _prices(broker, symbols):
    out = {}
    for s in symbols:
        out[s] = _safe_price(broker, s)
    return out


def _safe_price(broker, symbol):
    try:
        return float(broker.latest_price(symbol))
    except Exception as e:  # noqa: BLE001
        log.warning("price unavailable for %s: %s", symbol, e)
        return None


# --------------------------------------------------------------------------- #
# logging / printing
# --------------------------------------------------------------------------- #
def _log_run(as_of, mode, equity, signals, targets, held, prices, orders,
             placed, ew, spy, turnover, exposure, halted):
    ts = as_of.isoformat()
    placed_by = {s: oid for s, _, oid in placed}
    dec_cols = ["ts", "mode", "symbol", "score", "confidence", "rationale",
                "target_weight", "current_weight", "target_shares", "current_shares",
                "delta", "price", "order_id", "ok", "error"]
    order_by = {o.symbol: o for o in orders}
    for sym in sorted(set(signals) | set(targets) | set(held)):
        sig = signals.get(sym)
        o = order_by.get(sym)
        px = prices.get(sym) or 0.0
        cur = int(held.get(sym, 0))
        cur_w = (cur * px / equity) if equity else 0.0
        append_csv(config.AI_DECISIONS_LOG, {
            "ts": ts, "mode": mode, "symbol": sym,
            "score": round(sig.score, 4) if sig else "",
            "confidence": round(sig.confidence, 4) if sig else "",
            "rationale": (sig.rationale[:200] if sig else ""),
            "target_weight": round(targets.get(sym, 0.0), 4),
            "current_weight": round(cur_w, 4),
            "target_shares": (o.target_shares if o else (cur if sym in held else 0)),
            "current_shares": cur, "delta": (o.delta if o else 0),
            "price": round(px, 4),
            "order_id": placed_by.get(sym, ""),
            "ok": (sig.ok if sig else ""), "error": (sig.error if sig else ""),
        }, dec_cols)

    eq_cols = ["ts", "mode", "strategy_equity", "ew_universe_equity", "spy_equity",
               "gross_exposure", "n_positions", "turnover", "exposure_mult", "halted"]
    invested = sum(max(0, int(held.get(s, 0))) * (prices.get(s) or 0) for s in held)
    append_csv(config.AI_EQUITY_LOG, {
        "ts": ts, "mode": mode, "strategy_equity": round(equity, 2),
        "ew_universe_equity": round(ew, 2), "spy_equity": round(spy, 2),
        "gross_exposure": round(invested / equity, 4) if equity else 0.0,
        "n_positions": sum(1 for q in held.values() if q),
        "turnover": round(turnover, 4), "exposure_mult": round(exposure, 3),
        "halted": halted,
    }, eq_cols)


def _print_summary(as_of, mode, equity, signals, targets, held, prices, orders,
                   skipped, ew, spy, exposure, halted, notes, strategy):
    print(f"\n=== AI PORTFOLIO REBALANCE [{mode}]  {as_of.isoformat(timespec='seconds')} ===")
    print(f"equity ${equity:,.2f}  |  exposure x{exposure:.2f}"
          + ("  [KILL SWITCH ACTIVE]" if halted else ""))
    print(f"LLM: {getattr(strategy, 'llm', None) and strategy.llm.name}")
    print(f"\n  {'sym':<7}{'score':>7}{'conf':>6}{'tgt_w':>8}{'cur_sh':>8}{'tgt_sh':>8}  rationale")
    for sym in sorted(signals, key=lambda s: -(signals[s].score)):
        s = signals[sym]
        o = next((x for x in orders if x.symbol == sym), None)
        tgt_sh = o.target_shares if o else (int(held.get(sym, 0)) if targets.get(sym) else 0)
        print(f"  {sym:<7}{s.score:>7.2f}{s.confidence:>6.2f}{targets.get(sym, 0):>8.1%}"
              f"{int(held.get(sym, 0)):>8}{tgt_sh:>8}  {s.rationale[:70]}")
    if notes:
        print("  constructor:", "; ".join(notes))
    if orders:
        print("\n  orders:")
        for o in orders:
            tag = " (NEW)" if o.new_entry else ""
            print(f"    {o.side} {abs(o.delta)} {o.symbol} -> {o.target_shares} sh "
                  f"({o.target_weight:.1%}){tag}"
                  + ("  [DRY: not sent]" if mode == "dry" else ""))
    else:
        print("\n  no orders (targets already met)")
    if skipped:
        print("  skipped:", ", ".join(f"{s}({why})" for s, why in skipped))
    print(f"\n  benchmarks since start -> equal-weight ${ew:,.2f}   SPY ${spy:,.2f}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def run_stability():
    """Print the score-stability verdict per symbol from the decisions log."""
    from agents import stability
    if not os.path.exists(config.AI_DECISIONS_LOG):
        print(f"No decisions log yet at {config.AI_DECISIONS_LOG}. Run some rebalances first.")
        return
    by_symbol, runs = stability.load_recent_scores()
    results = stability.analyze(by_symbol)
    print(f"\n=== SCORE STABILITY (last {len(runs)} runs) ===")
    print(stability.format_table(results))
    flagged = stability.unstable_symbols(results)
    print("\n  unstable/flipped:", ", ".join(flagged) if flagged else "none")
    print("  (advisory — names that keep flipping sign are trading on noise)\n")


def run_report(out_png="results/ai/report.png"):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = config.AI_EQUITY_LOG
    if not os.path.exists(path):
        print(f"No equity log yet at {path}. Run at least one rebalance first.")
        return
    df = pd.read_csv(path, parse_dates=["ts"]).sort_values("ts")
    if df.empty:
        print("Equity log is empty."); return

    def dd(series):
        s = series.astype(float)
        return ((s - s.cummax()) / s.cummax()).min()

    strat0 = df["strategy_equity"].iloc[0]
    summary = {
        "runs": len(df),
        "start": df["ts"].iloc[0].date().isoformat(),
        "end": df["ts"].iloc[-1].date().isoformat(),
        "strategy_return_%": (df["strategy_equity"].iloc[-1] / strat0 - 1) * 100,
        "ew_return_%": (df["ew_universe_equity"].iloc[-1] / df["ew_universe_equity"].iloc[0] - 1) * 100,
        "spy_return_%": (df["spy_equity"].iloc[-1] / df["spy_equity"].iloc[0] - 1) * 100,
        "strategy_maxdd_%": dd(df["strategy_equity"]) * 100,
        "avg_turnover_%": df["turnover"].astype(float).mean() * 100,
    }
    print("\n=== FORWARD-TEST REPORT ===")
    for k, v in summary.items():
        print(f"  {k:<20}: {v:.2f}" if isinstance(v, float) else f"  {k:<20}: {v}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    ax1.plot(df["ts"], df["strategy_equity"], label="AI strategy", lw=1.8)
    ax1.plot(df["ts"], df["ew_universe_equity"], label="Equal-weight universe", lw=1.2, ls="--")
    ax1.plot(df["ts"], df["spy_equity"], label="SPY", lw=1.2, ls=":")
    ax1.set_ylabel("Equity ($)"); ax1.legend(loc="upper left"); ax1.grid(alpha=0.25)
    ax1.set_title("AI news portfolio vs benchmarks (forward test)")
    s = df["strategy_equity"].astype(float)
    ddc = (s - s.cummax()) / s.cummax() * 100
    ax2.fill_between(df["ts"], ddc, 0, color="crimson", alpha=0.35)
    ax2.set_ylabel("Drawdown (%)"); ax2.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"\n  chart -> {out_png}\n  data  -> {path}")


# --------------------------------------------------------------------------- #
# wiring / CLI
# --------------------------------------------------------------------------- #
def confirm_live():
    print("\n*** LIVE MODE — THIS TRADES REAL MONEY ***")
    phrase = "trade real money"
    if input(f'Type "{phrase}" to proceed: ').strip().lower() != phrase:
        raise SystemExit("Confirmation failed. Aborting.")


def build(args):
    from broker.alpaca_broker import AlpacaBroker
    from agents.llm import build_llm
    from agents.news_portfolio import NewsPortfolioStrategy
    from portfolio.inventory import Inventory
    from signals.hype import HypeTracker

    broker = AlpacaBroker(paper=(args.mode != "live"))
    llm = build_llm(provider=args.provider, model=args.model,
                    retries=config.AI_LLM_RETRIES, timeout=config.AI_LLM_TIMEOUT) \
        if args.provider != "stub" else build_llm(provider="stub")
    strategy = NewsPortfolioStrategy(llm=llm, universe=config.AI_UNIVERSE,
                                     exposure_pass=config.AI_EXPOSURE_PASS)
    limits = PortfolioLimits.from_config()
    killswitch = KillSwitch()
    inventory = Inventory(broker=broker)
    hype = HypeTracker()
    return broker, strategy, limits, killswitch, inventory, hype


def main():
    p = argparse.ArgumentParser(description="AI news-driven paper portfolio (forward test).")
    p.add_argument("command", nargs="?", default="run",
                   choices=["run", "report", "stability"])
    p.add_argument("--mode", default="paper", choices=["dry", "paper", "live"])
    p.add_argument("--provider", default=config.AI_LLM_PROVIDER,
                   help="LLM provider: gemini (default) or stub (offline)")
    p.add_argument("--model", default=config.AI_MODEL)
    p.add_argument("--once", action="store_true", help="run a single rebalance and exit")
    p.add_argument("--loop", type=int, metavar="SECONDS", help="loop every N seconds")
    p.add_argument("--reset-state", action="store_true", help="clear the kill-switch state")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.command == "report":
        run_report(); return

    if args.command == "stability":
        run_stability(); return

    if args.reset_state:
        KillSwitch().reset(); print("Kill-switch state reset.")
    if args.mode == "live":
        confirm_live()

    broker, strategy, limits, killswitch, inventory, hype = build(args)
    acct = broker.account_summary()
    print(f"Account: {acct['mode']}  equity=${acct['equity']:,.2f}  "
          f"buying_power=${acct['buying_power']:,.2f}")
    if acct["blocked"]:
        raise SystemExit("Trading is blocked on this account.")
    universe = [s.upper() for s in config.AI_UNIVERSE]

    def _cycle():
        run_once(broker, strategy, limits, killswitch, universe, args.mode,
                 hype_tracker=hype, inventory=inventory)

    if args.loop:
        print(f"Looping every {args.loop}s. Ctrl-C to stop.")
        try:
            while True:
                _cycle(); time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        _cycle()


if __name__ == "__main__":
    main()
