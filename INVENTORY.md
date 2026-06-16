# Portfolio Inventory / State (`portfolio/inventory.py`)

One reliable interface that reports **exactly what is held** at all times, over two
backends (live/paper broker and the backtest engine), so the strategy, the Phase 3
AI, and the risk code never disagree about positions.

> Shared INPUT for the Phase 3 AI. This module reports and logs only — it places no
> orders and is not wired into trading.

## Source-of-truth rule (critical)

- In **paper/live, Alpaca is authoritative** for quantities and cost basis. The
  inventory `sync()`s from the broker and **never overrides it**.
- The local metadata store holds only what Alpaca doesn't keep (entry date,
  strategy/rationale tag, target weight, stop level, hype-at-entry) plus an
  optional `expected_qty` used **only** for reconciliation/sanity checks — never
  written back to the broker.
- In **backtest**, the engine's simulated positions are the source; `SimBroker`
  presents that state through the same interface.

## Interface

```python
from portfolio.inventory import Inventory
from broker.alpaca_broker import AlpacaBroker

inv = Inventory(broker=AlpacaBroker(paper=True)).sync()

inv.holdings()       # {symbol: {qty, avg_cost, cost_basis, last_price, market_value,
                     #           weight, unrealized_pl, unrealized_pl_pct, side, metadata}}
inv.totals()         # equity, cash, gross/net exposure (+%), position_count, largest_weight, ...
inv.get("AAPL")      # that symbol's holding (zeroed if not held)
inv.reconcile()      # [divergences] vs expected_qty; logs & flags, never overrides
inv.snapshot(note)   # append timestamped holdings+totals to the history log
```

### Fields per holding

| field | meaning |
|---|---|
| `qty` | signed share count (+ long / − short), from the broker |
| `avg_cost` | Alpaca `avg_entry_price` |
| `cost_basis` | `qty × avg_cost` |
| `last_price` | broker `current_price` (or derived from market value) |
| `market_value` | `qty × last_price` |
| `weight` | `market_value / equity` |
| `unrealized_pl` | `market_value − cost_basis` (correct for long & short) |
| `unrealized_pl_pct` | `unrealized_pl / |cost_basis|` |
| `metadata` | merged local metadata (entry date, tag, stop, hype-at-entry, …) |

### Worked example (verification fixture)

Broker reports **326 AAPL @ $291.84**, live price **$320.00**, account equity **$200,000**:

```
cost_basis    = 326 × 291.84      = $95,139.84
market_value  = 326 × 320.00      = $104,320.00
unrealized_pl = 104,320 − 95,139.84 = $9,180.16   (+9.65%)
weight        = 104,320 / 200,000   = 52.16%
```

`tests/test_inventory.py::TestAAPLFixture` asserts exactly these numbers against a mock broker.

## Metadata store

SQLite (`config.PORTFOLIO_DB`), keyed by symbol, surviving restarts. Allowed fields:
`entry_date, strategy_tag, rationale, target_weight, stop_level, hype_at_entry,
expected_qty`. `set()` merges (doesn't clobber); unknown fields raise.

```python
inv.meta.set("AAPL", entry_date="2026-06-15", strategy_tag="supertrend",
             stop_level=280.0, hype_at_entry=0.71, expected_qty=326)
```

## Reconciliation (flag, never override)

`reconcile(expected=None)` compares locally **expected** quantities (passed in, or
each symbol's stored `expected_qty`) against the broker's **actual** quantities and
returns/logs divergences — without touching the broker:

- `qty_mismatch` — expected and actual both nonzero but differ (e.g. a partial fill)
- `missing_position` — expected nonzero, actual zero
- `untracked_position` — held at the broker with no expectation (manual trade / drift)

## Snapshots (forward-test record)

`snapshot()` appends a timestamped record to both a flat CSV
(`config.PORTFOLIO_HISTORY_CSV`, one row per holding + a `__TOTALS__` row) and a
`snapshots` table in the SQLite DB (full JSON, queryable by time). Exact UTC
timestamps. This builds the time series the forward test and the AI will read.

## Backtest adapter

```python
from portfolio.inventory import from_backtest
inv = from_backtest("AAPL", qty=100, avg_cost=150.0, last_price=165.0, cash=5_000.0)
inv.holdings()["AAPL"]["market_value"]   # 16,500 — same interface as live
```

`SimBroker` / `from_backtest()` populate the identical interface from the engine's
simulated state, so tests and backtests use the same `Inventory` API as live.

## CLI

```bash
python show_state.py inventory            # holdings synced from Alpaca (paper)
python show_state.py inventory --live     # live account
python show_state.py inventory --snapshot # also append to the history log
```

Without `ALPACA_KEY`/`ALPACA_SECRET` it explains how to set them rather than failing.

## How the Phase 3 AI will consume this

The AI reads `holdings()`/`totals()` to know current exposure and per-name weights
before proposing changes, uses `get(symbol)` to see entry rationale/stop/hype-at-entry
metadata for context, and relies on `reconcile()` to refuse to act when local
expectations and broker reality disagree. Risk sizing will read `totals()` for gross/net
exposure. **Not wired in yet** — this phase only provides the state.
