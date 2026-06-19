"""
Dynamic-universe DISCOVERY CLI — eyeball the screen and the daily vetting offline.

  python discover.py --screen-only        # just print the deterministic screener output
  python discover.py --mode dry           # screener -> LLM eval -> gate (NO store writes)
  python discover.py --mode dry --rebalance  # then a dry rebalance within the new universe

Step 1 of the build process: print a sample day's REAL candidates so they can be
eyeballed before anything trades. Step 3: a --mode dry day showing discovery -> vetted
additions -> rebalance within the new universe, placing no orders.

Requires Alpaca PAPER keys in the environment (the screener reads real market data).
The LLM is optional: with no GEMINI/COHERE/OPENAI key it falls back to the offline stub.
"""

from __future__ import annotations

import argparse
import logging

import config


def main():
    p = argparse.ArgumentParser(description="Run/preview dynamic-universe discovery.")
    p.add_argument("--mode", default="dry", choices=["dry", "paper", "live"],
                   help="dry (default): compute + print, write nothing")
    p.add_argument("--provider", default=config.AI_LLM_PROVIDER, help="gemini | stub")
    p.add_argument("--model", default=config.AI_MODEL)
    p.add_argument("--screen-only", action="store_true",
                   help="only run the deterministic screener and print candidates")
    p.add_argument("--max-assets", type=int, default=None,
                   help="override SCREEN_MAX_ASSETS_SCANNED (smaller = faster sample)")
    p.add_argument("--rebalance", action="store_true",
                   help="after discovery, run a DRY rebalance within the new universe")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    from service.secrets import maybe_load_ssm
    maybe_load_ssm()

    from broker.alpaca_broker import AlpacaBroker
    from signals.hype import HypeTracker
    from universe.screener import ScreenConfig, screen, _print_table
    from universe.store import UniverseStore

    cfg = ScreenConfig.from_config()
    if args.max_assets is not None:
        cfg.max_assets_scanned = args.max_assets

    broker = AlpacaBroker(paper=(args.mode != "live"))
    hype = HypeTracker()

    if args.screen_only:
        _print_table(screen(broker=broker, hype=hype, cfg=cfg))
        return

    from agents.llm import build_llm
    from portfolio.inventory import Inventory
    from universe.discovery import run_discovery

    llm = build_llm(provider=args.provider, model=args.model) \
        if args.provider != "stub" else build_llm(provider="stub")
    store = UniverseStore()
    inventory = Inventory(broker=broker)

    print(f"\nUniverse before: {len(store.symbols())} symbols "
          f"({len(store.dynamic_symbols())} dynamic)  LLM={llm.name}  mode={args.mode}")
    res = run_discovery(broker=broker, store=store, llm=llm, hype=hype,
                        inventory=inventory, cfg=cfg, mode=args.mode)

    _print_table(res["candidates"])
    print(f"\n=== GATE DECISIONS (mode={args.mode}) ===")
    if res["admitted"]:
        for a in res["admitted"]:
            print(f"  ADMIT  {a['symbol']:<7} [{a['tier']}] conv={a['conviction']:+.2f} "
                  f"reasons={'|'.join(a['reasons'])}  {a['rationale'][:60]}")
    else:
        print("  (none admitted)")
    if res["evicted"]:
        print("  evicted (universe full):", ", ".join(res["evicted"]))
    rej = {}
    for sym, why in res["rejected"]:
        rej.setdefault(why.split(":")[0], []).append(sym)
    if rej:
        print("  rejected:", "; ".join(f"{why} x{len(syms)}" for why, syms in rej.items()))
    if args.mode == "dry":
        print("\n  [DRY] nothing written to the universe store.")
    print(f"\nUniverse after: {len(store.symbols())} symbols "
          f"({len(store.dynamic_symbols())} dynamic)")

    if args.rebalance:
        from agents.news_portfolio import NewsPortfolioStrategy
        from portfolio.risk import PortfolioLimits, KillSwitch, SpeculativeLimits
        from live_portfolio import run_once
        print("\n=== DRY REBALANCE within the (preview) universe ===")
        strat = NewsPortfolioStrategy(llm=llm, universe=store.symbols(),
                                      exposure_pass=config.AI_EXPOSURE_PASS)
        run_once(broker, strat, PortfolioLimits.from_config(), KillSwitch(),
                 store.symbols(), "dry", hype_tracker=hype, inventory=inventory,
                 tier_map=store.tier_map(), spec_limits=SpeculativeLimits.from_config())


if __name__ == "__main__":
    main()
