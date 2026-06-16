"""
Print the current portfolio inventory and the current hype ranking.

These are shared INPUTS for the Phase 3 AI — this CLI only reports them; it places
no orders.

Examples
--------
    python show_state.py                      # both inventory + hype (paper)
    python show_state.py inventory            # holdings synced from Alpaca
    python show_state.py hype --watchlist AAPL,MSFT,TSLA
    python show_state.py both --snapshot      # also append to the history logs

Inventory needs Alpaca keys (ALPACA_KEY/ALPACA_SECRET); without them it explains
how to set them. Hype uses free sources and degrades gracefully if any is missing.
"""

import argparse
import logging

import config


def _print_inventory(args):
    from portfolio.inventory import Inventory
    try:
        from broker.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(paper=not args.live)
    except Exception as e:  # noqa: BLE001
        print(f"\n[inventory] cannot reach Alpaca: {e}")
        print("            set ALPACA_KEY / ALPACA_SECRET (paper keys work) and retry.")
        return

    inv = Inventory(broker=broker).sync()
    totals = inv.totals()
    holdings = inv.holdings()

    print("\n=== INVENTORY ({mode}) ===".format(mode=totals["mode"]))
    print(f"equity ${totals['equity']:,.2f}   cash ${totals['cash']:,.2f}   "
          f"positions {totals['position_count']}   "
          f"gross {totals['gross_exposure_pct']:.0%}   net {totals['net_exposure_pct']:.0%}   "
          f"largest {totals['largest_weight']:.0%}")
    if not holdings:
        print("  (no open positions)")
    else:
        print(f"  {'symbol':<8}{'qty':>10}{'avg_cost':>11}{'last':>10}"
              f"{'mkt_value':>13}{'weight':>8}{'unreal_pl':>13}{'pl%':>8}{'managed':>9}")
        for sym, h in sorted(holdings.items(), key=lambda kv: -abs(kv[1]['weight'])):
            print(f"  {sym:<8}{h['qty']:>10.2f}{h['avg_cost']:>11.2f}{h['last_price']:>10.2f}"
                  f"{h['market_value']:>13,.2f}{h['weight']:>8.1%}"
                  f"{h['unrealized_pl']:>13,.2f}{h['unrealized_pl_pct']:>8.1%}"
                  f"{('yes' if h['managed'] else 'no'):>9}")

    divergences = inv.reconcile()
    if divergences:
        print(f"  reconcile: {len(divergences)} divergence(s) vs expected -> {divergences}")
    if args.snapshot:
        inv.snapshot(note="cli")
        print(f"  snapshot appended -> {inv.history_path}")


def _print_hype(args):
    from signals.hype import HypeTracker
    watchlist = ([s.strip().upper() for s in args.watchlist.split(",") if s.strip()]
                 if args.watchlist else
                 (config.HYPE_DISCOVERY if args.discovery else config.HYPE_WATCHLIST))
    tracker = HypeTracker()
    print(f"\n=== HYPE RANKING ({len(watchlist)} symbols) ===")
    rows = tracker.snapshot(watchlist) if args.snapshot else tracker.rank(watchlist)
    print(f"  {'rank':<5}{'symbol':<8}{'score':>7}   components (used)")
    for i, r in enumerate(rows, 1):
        score = "  n/a" if r["score"] != r["score"] else f"{r['score']:.3f}"
        comps = ", ".join(f"{k}={v:.2f}" for k, v in r["components"].items())
        miss = f"  [missing: {','.join(r['missing'])}]" if r["missing"] else ""
        print(f"  {i:<5}{r['symbol']:<8}{score:>7}   {comps}{miss}")
    if args.snapshot:
        print(f"  snapshot appended -> {tracker.history_csv}")


def main():
    p = argparse.ArgumentParser(description="Show portfolio inventory and hype ranking.")
    p.add_argument("what", nargs="?", default="both", choices=["inventory", "hype", "both"])
    p.add_argument("--live", action="store_true", help="use the live account (default paper)")
    p.add_argument("--watchlist", help="comma-separated symbols for hype (overrides config)")
    p.add_argument("--discovery", action="store_true",
                   help="use the broader HYPE_DISCOVERY list instead of the watchlist")
    p.add_argument("--snapshot", action="store_true", help="append results to history logs")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.what in ("inventory", "both"):
        _print_inventory(args)
    if args.what in ("hype", "both"):
        _print_hype(args)
    print()


if __name__ == "__main__":
    main()
