# Always-On Service (`live_service.py`) — Phase 3, Step 1

A single long-running process that turns the run-once / sleep-loop batch runner into a
supervised, **two-cadence**, streaming service. It runs on **live data only** (never the
backtest CSV) and keeps memory tiny so it fits a 1 GB box.

> **Not investment advice.** Forward experiment. Paper by default. Step 1 is the service
> itself; deployment is Step 2.

## Architecture

```
                         ┌──────────────────────────── Supervisor (asyncio) ───────────────────────────┐
 Alpaca websocket  ──►   │  BarBuffer (rolling, in-memory, last N min bars/symbol, thread-safe)         │
 (IEX minute bars)       │                                                                              │
                         │  FAST loop  (~60s, market-gated)   SLOW loop  (~60min, market-gated)         │
                         │   • deterministic scan (no LLM)     • get_info news → Gemini → constructor    │
                         │     reuse registry strategy         • reconcile → orders (live_portfolio)     │
                         │   • kill-switch → go flat           • then protective.reconcile()             │
                         │   • protective.reconcile()                                                   │
                         │                                    SYNC loop (~15min, ungated)               │
                         │                                     • Alpaca-authoritative inventory sync     │
                         └──────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                         Server-side PROTECTIVE resting orders at Alpaca (GTC)
                         — fire at the exchange even if this process is down.
```

Everything reuses existing code: `broker/alpaca_broker.py`, `risk/manager.py`,
`portfolio/constructor.py`, `portfolio/inventory.py`, `agents/news_portfolio.py`, the
`get_info` news tool, `strategies/registry`, and `live_portfolio.run_once`. The backtest
engine is untouched.

## The two cadences

| Loop | Default | Gated by market hours | What it does |
|---|---|---|---|
| **Fast** | every `SERVICE_SCAN_INTERVAL_SEC` (60s) | yes | deterministic pattern/risk scan over the buffer (no LLM) + kill-switch check + **protective-order reconciliation** |
| **Slow** | every `SERVICE_REBALANCE_INTERVAL_MIN` (60min) | yes | the existing AI news rebalance (LLM scoring → constructor → orders), then immediately reconciles protective orders for new entries |
| **Sync** | every `SERVICE_INVENTORY_SYNC_MIN` (15min) | no | re-syncs the inventory from Alpaca (source of truth) |

The fast loop is **detection + backstop**, *not* the primary exit path. LLM calls only
happen on the slow loop, spaced via the existing `AI_LLM_SLEEP` / 429-backoff, and the
"429 → score 0 / no change" guardrails stay in force.

## Protective resting orders (the real "act at any moment")

Whenever a **managed** long position exists, the service keeps a server-side SELL order
resting at Alpaca (GTC), so the exit fires at the exchange even if the bot or the box is
down. Configurable via `PROTECT_*`:

- **stop** (default): stop-loss at `entry × (1 − PROTECT_STOP_PCT)` (RiskManager distance).
- **trailing**: set `PROTECT_TRAILING_PCT` for a trailing stop instead.
- **OCO bracket**: set `PROTECT_BRACKET_OCO=True` with `PROTECT_TAKE_PROFIT_PCT` for a
  take-profit limit + stop, one-cancels-the-other.

`reconcile()` is **idempotent**: it places an order only when a managed position lacks
matching protection, skips when already covered (no double-submission), and cancels +
re-places when a resize makes the resting qty stale. Unmanaged/discovered positions are
**walled off** (never touched), consistent with the inventory model. In `--mode dry` it
logs the intended action and places nothing.

## Market-hours behavior

Gating uses Alpaca's authoritative clock for **regular hours**. With `--extended-hours`
(or `SERVICE_EXTENDED_HOURS=True`) the fast/slow loops also run in the pre-/post-market
windows (a NY-time heuristic). When closed, gated loops idle (just sleeping) — **no LLM
calls, minimal CPU** — and resume on the next open. If the clock call fails, the service
fails **closed** (safe). The sync loop runs regardless.

## Safety

- **Modes:** `--mode dry` (compute + log, **no orders**) · `paper` (default) · `live`
  (existing triple gate: `--mode live` **and** `ALPACA_ALLOW_LIVE=yes` **and** a typed
  confirmation).
- **Drawdown kill switch:** persisted across runs (`results/ai/portfolio_state.json`).
  The fast loop checks real equity each tick; on trip it **goes flat** (cancels protective
  orders + market-closes managed longs) and the slow loop stops opening.
- **Exception isolation:** an exception in any loop iteration is logged with a full trace
  and swallowed — it never kills the task, its peers, or the process.
- **Graceful shutdown** (SIGINT/SIGTERM): stops the timed loops and the stream but
  **leaves resting protective orders in place** and opens nothing on the way out.
- **Auto-reconnect:** the stream self-heals; the SDK manages in-session websocket
  reconnects and a watchdog restarts `run()` if it ever exits.

## Config (in `config.py`)

```
SERVICE_SCAN_INTERVAL_SEC = 60        SERVICE_REBALANCE_INTERVAL_MIN = 60
SERVICE_INVENTORY_SYNC_MIN = 15       SERVICE_EXTENDED_HOURS = False
SERVICE_BUFFER_BARS = 240             SERVICE_FEED = "iex"
SERVICE_FAST_STRATEGY = "supertrend"  SERVICE_FAST_MIN_BARS = 30
PROTECT_ENABLED = True                PROTECT_STOP_PCT = 0.08
PROTECT_TRAILING_PCT = None           PROTECT_TAKE_PROFIT_PCT = None
PROTECT_BRACKET_OCO = False
# universe, model, LLM spacing reuse the AI_* settings.
```

## Run locally

```bash
set -a && source .env && set +a            # ALPACA_KEY/SECRET (+ GEMINI_API_KEY for real LLM)

# Dry run (no orders), real cadences:
python live_service.py --mode dry

# Paper trading (default):
python live_service.py --mode paper --scan-interval 60 --rebalance-interval 60

# Local demo when the market is closed (dev gate bypass; still no orders in dry):
python live_service.py --mode dry --provider stub --no-stream --force-open \
       --scan-interval 20 --rebalance-interval 1 --duration 120

# Live (real money) — triple gated:
ALPACA_ALLOW_LIVE=yes python live_service.py --mode live
```

Flags: `--scan-interval` (sec), `--rebalance-interval` (min), `--extended-hours`,
`--provider {gemini,stub}`, `--no-stream`, `--force-open` (dev gate bypass),
`--duration N` (run N seconds then stop — for demos/tests), `--reset-state`.

This is **not** a system scheduler and does not autostart — you launch it yourself, and a
clean Ctrl-C leaves your protective orders resting at the exchange.
```
