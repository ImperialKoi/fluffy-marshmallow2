# Dynamic Universe Discovery

> Once per day the bot **discovers** new tradable US stocks to add to its trading
> universe, biased toward high-risk/high-reward profiles — low-priced/penny names,
> high realized volatility, and high relative volume with low float ("high volume,
> low supply"). The existing AI rebalance then allocates within the expanded universe.
>
> Same design law as the rest of Phase 3: **the LLM is advisory; a deterministic
> layer has the final say.** Every existing risk control stays on. The dynamic
> universe can be turned off (pinned to the fixed list) with one config flag.

**Not investment advice. Penny / low-float names are dangerous (see Caveats §7).**

---

## 1. The one rule that makes this safe

**The LLM never generates tickers.** Discovery is driven by a *deterministic
screener over real market data*. The LLM only **evaluates / ranks** the screener's
real candidates (with news context), exactly like the existing per-symbol scorer.

Any ticker the LLM returns that is **not** in the screened, tradable candidate set is
**ignored** — structurally, because the gate iterates the *screener's* candidates and
attaches LLM scores to them, never the reverse. This prevents hallucinated, delisted,
or untradeable symbols from ever entering the universe.

---

## 2. Flow

```
DAILY (separate from the 60s scan and the 60min rebalance):

  screener  ─►  LLM evaluation  ─►  deterministic gate  ─►  persisted universe store
 (real data)    (advisory)         (FINAL SAY)              (survives restarts)
                                                                    │
                                                                    ▼
                                          existing hourly AI rebalance allocates
                                          within the expanded universe (constructor
                                          + risk layer, incl. the speculative sleeve)
```

| Stage | File | Role |
|-------|------|------|
| Screener | `universe/screener.py` | Deterministic candidate source over real Alpaca data |
| LLM eval | `universe/discovery.py` (`DiscoveryScorer`) | Reuses the news scorer; advisory inclusion ranking |
| Gate | `universe/discovery.py` (`gate`) | Tradability + liquidity + cap/churn; **final say** |
| Store | `universe/store.py` | Persisted membership ledger (symbol, tier, added_date, rationale) |
| Risk | `portfolio/risk.py` (`enforce_speculative_sleeve`) | Hard sleeve cap + tighter per-name cap |
| Wiring | `live_service.py` + `service/supervisor.py` | Daily discovery task; dynamic universe in the rebalance |
| CLI | `discover.py`, `python -m universe.screener` | Eyeball candidates / run a dry day |

---

## 3. The deterministic screener (`universe/screener.py`)

Candidates come **only** from here. Pipeline:

1. **Tradable universe.** Start from Alpaca's `get_all_assets` filtered to *active,
   tradable, US-equity* on a supported exchange (`SCREEN_EXCHANGES`). This is the
   tradability guarantee: many penny/OTC names are **not** on Alpaca and are
   auto-excluded right here.
2. **Per-symbol screens** over recent daily bars (`SCREEN_LOOKBACK_DAYS`):
   * **penny / high-risk** — last price `≤ SCREEN_PRICE_CAP` (default $5);
   * **high volatility** — stdev of daily returns `≥ SCREEN_MIN_VOLATILITY`;
   * **high relative volume** — today's volume vs trailing average `≥ SCREEN_RVOL_MIN`
     (the "high volume" half of "high volume, low supply");
   * **low supply** — float / shares outstanding `≤ SCREEN_MAX_FLOAT` via yfinance
     (**best-effort**, expect gaps; a missing float never drops a name unless
     `SCREEN_REQUIRE_FLOAT`, it just forgoes the low-float ranking bonus).
3. **Liquidity floor (HARD gate).** Average daily **dollar** volume must be
   `≥ SCREEN_MIN_DOLLAR_VOLUME`. This applies **even to penny picks** — never propose
   a name so thin that the bot's own order moves the price. A candidate that fails
   this is dropped no matter how exciting it looks.
4. **Hype feed.** The hype tracker's most-hyped names are folded in as an extra
   discovery feed (still subjected to the same tradability + liquidity gates).
5. **Rank & cap.** A composite score rewards volatility, relative volume, low float,
   cheapness, and hype; the top `SCREEN_DAILY_CANDIDATES` (default 20) are returned.

A name must clear the hard gates **price-cap → liquidity floor → at-least-one-hot-
signal** before it is even scored. Eyeball a real day:

```bash
python -m universe.screener --max-assets 600     # quick sample
python discover.py --screen-only                 # full screen, pretty table
```

---

## 4. LLM evaluation (`DiscoveryScorer`)

Reuses the existing Gemini news scorer (`agents/news_portfolio.py`): it pulls
`get_info` news/filings for each **screener** candidate, asks the LLM (batched, one
JSON call) to judge the **net asymmetric upside**, and parses the structured response.
It:

* is **advisory** — it never sizes a trade and never proposes a ticker;
* **audit-logs** every prompt + raw response + parsed result (dated JSONL);
* **spaces** calls (`DISCOVERY_LLM_SLEEP`) and **caps** the candidate count
  (`SCREEN_DAILY_CANDIDATES`) so it fits free-tier limits;
* **ignores any off-list ticker** the model emits (the batch parser only accepts the
  candidate set — the anti-hallucination contract, also enforced again at the gate).

A candidate is admitted only if its LLM **conviction = score × confidence** clears
`DISCOVERY_MIN_CONVICTION`.

---

## 5. The gate (deterministic, final say)

For each **screener** candidate, in ranked order, the gate:

1. **Re-confirms tradability** at admission time via `broker.get_asset` (an asset can
   be delisted/halted between the daily screen and now).
2. **Re-confirms the liquidity floor** and that price/data exist.
3. Requires **LLM conviction ≥ `DISCOVERY_MIN_CONVICTION`** (the model's vote).
4. **Respects the wall-off**: a symbol whose inventory metadata explicitly sets
   `managed=False` is treated as hand-excluded and skipped.
5. **Enforces the universe cap** (`UNIVERSE_MAX_SIZE`, default 40) so hourly scoring
   stays within free limits. When full, it **evicts the weakest/stalest** dynamic name
   (lowest admitted conviction, oldest `last_seen` as tie-break) — **never** a pinned
   core name and **never** a name with an open position.
6. **Tags** the admitted symbol with `risk_tier`, `added_date`, and the discovery
   `rationale` in the inventory metadata, and adds it to the persisted store.

Every decision (admitted/rejected, with reason) is logged to
`UNIVERSE_DISCOVERY_LOG`. A hallucinated ticker is impossible to admit: the gate only
ever iterates real screener candidates.

---

## 6. Risk model for the riskier universe

The riskier names enter a **speculative sleeve**. These controls **extend** the
existing risk layer — they only ever *tighten*, never weaken it.

* **Speculative sleeve (hard cap).** Combined equity weight across **all**
  speculative/penny names is capped at `SPEC_SLEEVE_PCT` (default 15%), so a blowup in
  thin names can't sink the book. Enforced deterministically by
  `enforce_speculative_sleeve` *after* the constructor, *before* orders.
* **Tighter per-name cap.** A speculative name is capped at `SPEC_MAX_WEIGHT`
  (default 5%) vs the core `AI_MAX_WEIGHT` (20%).
* **Tight stops AND take-profits.** Penny/speculative names get a tight
  `SPEC_STOP_PCT` (default 4%) stop and a tight `SPEC_TAKE_PROFIT_PCT` (default 8%)
  take-profit — vs the core 8% / 20%. This feeds **both**:
  * the server-side **protective / OCO** resting orders (`service/protective.py`),
    so each thin name rests at the exchange with a close stop *and* a close ceiling, and
  * the **deterministic exit engine** (`service/risk_exits.py`), which cuts a
    speculative loser / books a speculative gain much sooner — and still fires even if
    every LLM is offline.
  Tier is read per symbol from `risk_tier` in the inventory metadata.

**All existing controls remain in force**: the persisted drawdown **kill switch**,
**conviction gating**, the neutral **dead-band**, the **turnover cap**, the degraded-
signal guard, and the managed wall-off.

---

## 7. Cadence, integration & persistence

* A **daily** discovery task runs in `live_service.py` (cadence
  `--discovery-interval`, default `UNIVERSE_DISCOVERY_INTERVAL_HOURS = 24`), separate
  from the 60s fast scan and the 60min AI rebalance. It is market-hours gated and
  isolated — a failure is logged and never kills the process.
* The hourly **rebalance** reads the *current* universe from the store each run and
  passes the tier map + speculative limits into the existing constructor + risk layer.
* The **universe store** (`UNIVERSE_STORE_FILE`, JSON) persists dynamic symbols (tier,
  added_date, rationale, conviction, last_seen) so the universe **survives restarts**.
* **Pin it any time.** `UNIVERSE_DISCOVERY_ENABLED = False`, the `--no-discovery` flag,
  or an explicit `--universe A,B,C` pins the universe to the fixed list and skips
  discovery entirely (everything else behaves exactly as before).

```bash
# eyeball a sample day's real candidates (no trading)
python discover.py --screen-only

# full dry day: screener -> LLM eval -> gate (writes nothing), optionally a dry rebalance
python discover.py --mode dry
python discover.py --mode dry --rebalance

# always-on service WITH daily discovery (paper); or pin the universe
python live_service.py --mode paper
python live_service.py --mode paper --no-discovery
```

---

## 8. Honest caveats

* **Penny / low-float manipulation & slippage.** Low-priced, low-float names are the
  classic venue for pump-and-dump and violent gaps. They can gap *through* a stop
  (the stop becomes a market order at a worse price). The liquidity floor, the hard
  sleeve cap, the tighter per-name cap, and the tight stops/TPs are mitigations, **not
  guarantees**. Position sizes here are deliberately small.
* **Market-order fills are worse on thin names.** The bot trades market orders; on a
  thin book the realized fill can be materially worse than the last print. Measure the
  live-vs-expected gap on paper before trusting it.
* **yfinance float data is best-effort.** It is frequently missing or stale; a missing
  float never blocks a name (unless `SCREEN_REQUIRE_FLOAT`), so the "low supply" tilt
  is a *bonus signal*, not a hard gate.
* **Free IEX data is partial volume.** Relative-volume and dollar-volume are computed
  on the IEX feed (~a few % of consolidated volume); SIP (paid) is more accurate.
* **Newly discovered names** are priced via REST in the rebalance and protected by
  resting exchange orders, but the live minute-bar stream only subscribes to the
  startup universe — the in-memory deterministic crash/S-R checks warm up for a new
  name across a restart. The hard stop/TP (resting + entry-relative) still applies.
* **No backtest of the discovery signal exists.** This is a forward-test experiment on
  paper, gated behind every risk control above — not a proven edge.
