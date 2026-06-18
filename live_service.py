"""
Always-on AI trading SERVICE (Phase 3, Step 1) — a long-running, two-cadence runner.

Unlike `live_portfolio.py` (run-once / sleep-loop batch), this is one supervised process:
  * streams live minute bars into an in-memory buffer (no CSV; small memory),
  * FAST loop (~60s): deterministic pattern/risk scan + protective-order reconciliation,
  * SLOW loop (~60min): the existing AI news rebalance (get_info -> Gemini -> constructor),
  * server-side PROTECTIVE resting orders fire at the exchange even if this process is down,
  * market-hours gated; idle cleanly when closed.

Modes: --mode dry (compute + log, NO orders) | paper (default) | live (triple-gated).
The persisted drawdown kill switch stays active; on trip the fast loop goes flat.

This is NOT a system scheduler/autostart — you start it yourself. Examples:
    python live_service.py --mode dry
    python live_service.py --mode paper --scan-interval 60 --rebalance-interval 60
    python live_service.py --mode dry --provider stub --duration 120 --force-open  # local demo
"""

import argparse
import asyncio
import logging
import signal
from datetime import datetime, timezone

import config
from strategies.registry import build as build_strategy
from agents.llm import build_llm
from agents.news_portfolio import NewsPortfolioStrategy
from portfolio.risk import PortfolioLimits, KillSwitch
from portfolio.inventory import Inventory
from signals.hype import HypeTracker
from service.buffer import BarBuffer
from service.clock import MarketClock
from service.protective import ProtectiveOrderManager
from service.fast_scan import run_fast_scan
from service.supervisor import Supervisor
from live_portfolio import run_once, confirm_live

log = logging.getLogger("live_service")


class _AlwaysOpenClock:
    """Dev/testing gate bypass (clearly logged). No orders are placed in dry mode anyway."""
    def is_open(self):
        return True

    def status(self):
        return {"is_open": True, "forced": True}


def build_service(args):
    from broker.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(paper=(args.mode != "live"))
    universe = [s.upper() for s in (args.universe.split(",") if args.universe
                                    else config.AI_UNIVERSE)]

    llm = build_llm(provider=args.provider, model=args.model,
                    retries=config.AI_LLM_RETRIES, timeout=config.AI_LLM_TIMEOUT) \
        if args.provider != "stub" else build_llm(provider="stub")
    strategy = NewsPortfolioStrategy(llm=llm, universe=universe,
                                     exposure_pass=config.AI_EXPOSURE_PASS)
    limits = PortfolioLimits.from_config()
    killswitch = KillSwitch()
    inventory = Inventory(broker=broker)
    hype = HypeTracker()
    protective = ProtectiveOrderManager()
    from service.risk_exits import ExitSettings
    exit_settings = ExitSettings.from_config()
    fast_strategy = build_strategy(config.SERVICE_FAST_STRATEGY)
    buffer = BarBuffer(maxlen=config.SERVICE_BUFFER_BARS)

    clock = _AlwaysOpenClock() if args.force_open else \
        MarketClock(broker, extended_hours=(args.extended_hours or config.SERVICE_EXTENDED_HOURS))

    stream = None
    if not args.no_stream:
        from service.stream import AlpacaBarStream
        stream = AlpacaBarStream(buffer, universe, feed=config.SERVICE_FEED)

    return dict(broker=broker, universe=universe, strategy=strategy, limits=limits,
                killswitch=killswitch, inventory=inventory, hype=hype,
                protective=protective, exit_settings=exit_settings,
                fast_strategy=fast_strategy, buffer=buffer,
                clock=clock, stream=stream)


def make_tasks(svc, mode):
    broker, universe = svc["broker"], svc["universe"]
    inventory, killswitch = svc["inventory"], svc["killswitch"]
    protective, buffer = svc["protective"], svc["buffer"]
    exit_settings = svc["exit_settings"]

    def fast_fn():
        res = run_fast_scan(buffer=buffer, broker=broker, inventory=inventory,
                            killswitch=killswitch, protective=protective,
                            fast_strategy=svc["fast_strategy"], universe=universe,
                            mode=mode, min_bars=config.SERVICE_FAST_MIN_BARS,
                            exit_settings=exit_settings)
        placed = [a for a in res["protective"] if a.get("action") in ("placed", "would_place")]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log.info("[FAST %s] halted=%s signals=%d exits=%d protective:%d (new/intended=%d) flat=%d",
                 ts, res["halted"], len(res["signals"]), len(res.get("exits", [])),
                 len(res["protective"]), len(placed), len(res["flat"]))
        for e in res.get("exits", []):
            log.warning("    EXIT %s %s qty=%s reason=%s last=%s levels=%s",
                        e.get("action"), e["symbol"], e.get("qty"), e.get("reason"),
                        e.get("last"), e.get("levels"))
        for a in placed:
            log.info("    protective %s %s qty=%s stop=%s kind=%s",
                     a["action"], a["symbol"], a.get("qty"), a.get("stop_price"), a.get("kind"))

    def slow_fn():
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log.info("[SLOW %s] AI rebalance starting (%s)", ts, svc["strategy"].llm.name)
        run_once(broker, svc["strategy"], svc["limits"], killswitch, universe, mode,
                 hype_tracker=svc["hype"], inventory=inventory)
        # protect any freshly opened/resized positions immediately (don't wait a fast tick)
        protective.reconcile(broker, inventory, mode)
        # advisory score-stability check over recent runs (logged, non-blocking)
        try:
            from agents import stability
            by_sym, runs = stability.load_recent_scores()
            flagged = stability.unstable_symbols(stability.analyze(by_sym))
            if flagged:
                log.warning("[STABILITY] scores flipping sign over last %d runs: %s "
                            "(trading on noise — review before trusting)", len(runs), flagged)
        except Exception as e:  # noqa: BLE001
            log.debug("stability check skipped: %s", e)

    def sync_fn():
        try:
            inventory.sync()
            t = inventory.totals()
            log.info("[SYNC] equity=$%s positions=%d gross=%.0f%%",
                     f"{t['equity']:,.2f}", t["position_count"], t["gross_exposure_pct"] * 100)
        except Exception as e:  # noqa: BLE001
            log.warning("inventory sync failed: %s", e)

    return fast_fn, slow_fn, sync_fn


async def _run(supervisor, duration):
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, supervisor.request_stop)
        except (NotImplementedError, RuntimeError):
            pass
    if duration:
        await supervisor.run_for(duration)
    else:
        await supervisor.run()


def main():
    p = argparse.ArgumentParser(description="Always-on AI trading service (two cadences).")
    p.add_argument("--mode", default="paper", choices=["dry", "paper", "live"])
    p.add_argument("--provider", default=config.AI_LLM_PROVIDER, help="gemini | stub")
    p.add_argument("--model", default=config.AI_MODEL)
    p.add_argument("--universe", default=None,
                   help="comma-separated symbols to override config.AI_UNIVERSE")
    p.add_argument("--scan-interval", type=int, default=config.SERVICE_SCAN_INTERVAL_SEC)
    p.add_argument("--rebalance-interval", type=int, default=config.SERVICE_REBALANCE_INTERVAL_MIN,
                   help="minutes between AI rebalances")
    p.add_argument("--extended-hours", action="store_true")
    p.add_argument("--no-stream", action="store_true", help="don't open the websocket (e.g. demo)")
    p.add_argument("--force-open", action="store_true",
                   help="DEV: bypass market-hours gating (no orders are placed in dry mode)")
    p.add_argument("--duration", type=int, help="run N seconds then stop (demo/testing)")
    p.add_argument("--reset-state", action="store_true", help="clear the kill-switch state")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    # On EC2 (env-gated), pull secrets from SSM into the environment via the instance
    # IAM role BEFORE the broker/LLM read them. No-op locally unless TRADINGBOT_USE_SSM.
    from service.secrets import maybe_load_ssm
    maybe_load_ssm()

    if args.reset_state:
        KillSwitch().reset(); print("Kill-switch state reset.")
    if args.mode == "live":
        confirm_live()

    svc = build_service(args)
    acct = svc["broker"].account_summary()
    print(f"Account: {acct['mode']}  equity=${acct['equity']:,.2f}  mode={args.mode}")
    if acct["blocked"]:
        raise SystemExit("Trading is blocked on this account.")

    # startup inventory sync (Alpaca-authoritative)
    try:
        svc["inventory"].sync()
    except Exception as e:  # noqa: BLE001
        log.warning("startup inventory sync failed: %s", e)

    # warm the bar buffer from REST history so the fast scan has data immediately
    # (otherwise the deterministic scan is blind until the websocket fills ~30 bars)
    try:
        from service.stream import warm_buffer
        warm_buffer(svc["buffer"], svc["universe"], n_bars=config.SERVICE_WARMUP_BARS,
                    feed=config.SERVICE_FEED)
    except Exception as e:  # noqa: BLE001
        log.warning("buffer warm-up skipped: %s", e)

    if args.force_open:
        log.warning("FORCE-OPEN: market-hours gating bypassed (dev/demo).")
    log.info("clock status: %s", svc["clock"].status())

    fast_fn, slow_fn, sync_fn = make_tasks(svc, args.mode)
    supervisor = Supervisor(
        clock=svc["clock"], scan_interval=args.scan_interval,
        rebalance_interval=args.rebalance_interval * 60,
        sync_interval=config.SERVICE_INVENTORY_SYNC_MIN * 60,
        fast_fn=fast_fn, slow_fn=slow_fn, sync_fn=sync_fn, stream=svc["stream"])

    print(f"Service up: fast={args.scan_interval}s  slow={args.rebalance_interval}min  "
          f"protective={svc['protective'].s.kind()}  stream={'on' if svc['stream'] else 'off'}")
    try:
        asyncio.run(_run(supervisor, args.duration))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
