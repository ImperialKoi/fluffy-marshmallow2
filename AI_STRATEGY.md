# Phase 3 — AI News-Driven Portfolio (Forward Test)

> **This is a live forward experiment, not a proven edge.** There is **no historical
> backtest** of the news signal — it depends on real-time news that isn't in the
> bundled dataset. The bar is beating an equal-weight buy-and-hold of the same
> universe **and** SPY over weeks of paper trading. It may not. Nothing here claims
> an edge; the accumulating paper record is the only evidence. **Not investment advice.**

## Decision flow

Per scheduled rebalance (default once daily), over a configurable universe of
liquid US large caps (`config.AI_UNIVERSE`, default 15, min 10):

1. **Gather** — for each symbol, `get_info(symbol, as_of=now)` pulls recent news,
   SEC filings, and analyst data. `as_of=now` guarantees no future leakage.
2. **Score (LLM, advisory)** — the material is summarized and sent to the LLM, which
   must return **strict JSON** per symbol: `{ticker, score∈[-1,1], confidence∈[0,1],
   rationale}`. Positive = bullish.
3. **Exposure pass (optional)** — if `AI_EXPOSURE_PASS`, the LLM also returns a
   whole-market risk-on/off multiplier ∈[0,1] to scale total exposure.
4. **Construct (deterministic)** — `portfolio/constructor.py` ranks the positive
   names, keeps the top `max_positions`, weights them equal- or confidence-weighted
   within an invested budget of `min(max_gross, 1−min_cash) × exposure`, caps each at
   `max_weight` (excess → cash), and scales down if fewer than `min_positions` names
   qualify. Long-only; the rest is cash. **The LLM never sizes a trade.**
5. **Risk (deterministic, always on)** — `portfolio/risk.py`: per-name/gross/cash
   limits plus a drawdown **kill switch** on real paper equity, persisted across runs.
   When tripped, the book goes flat and no new positions open until reset.
6. **Reconcile & execute** — target weights → target shares from account equity,
   diffed against current Alpaca positions; market orders for the differences (the
   single-symbol reconciliation, looped over the basket). Paper by default.

```
get_info → LLM scores (advisory) → constructor (weights) → risk (limits+killswitch) → reconcile → orders
```

## LLM guardrails (critical)

- **Structured JSON only**, schema + numeric ranges validated on every response.
- **Out-of-universe tickers ignored** — never acted on.
- **Fail safe:** malformed/empty/timeout/refusal/missing-news ⇒ that symbol defaults
  to **score 0 / no change**. A bad LLM response never crashes the run or places a
  wild order. (No news ⇒ the LLM isn't even called.)
- **Full audit trail:** every call's prompt, raw response, and parsed result is
  appended to a dated JSONL file in `results/ai_audit/` — you can see exactly what the
  AI decided and why.
- The LLM is **advisory**; the constructor and risk layer are deterministic and have
  final say.

## Provider

`agents/llm.py` wraps Gemini behind a thin `LLM` interface (swappable). It prefers the
new `google-genai` SDK and falls back to legacy `google-generativeai`. Key from
`GEMINI_API_KEY`. Default model `config.AI_MODEL` (`gemini-2.5-flash`; verify
availability for your key/SDK). **Without a key**, the harness uses an offline
`StubLLM` (a tiny deterministic lexical sentiment model) — clearly labeled, **not the
real model** — so the pipeline and tests still run end-to-end.

## Running it

```bash
cp .env.example .env       # set ALPACA_KEY/SECRET and GEMINI_API_KEY
set -a && source .env && set +a

# 1) Dry run: compute + log + audit, NO orders (cheap; eyeball decisions first)
python live_portfolio.py --mode dry --once

# 2) Paper trade (default), once per scheduled run (pair with cron)
python live_portfolio.py --mode paper --once
python live_portfolio.py --mode paper --loop 86400      # or self-loop daily

# Offline pipeline check (no Gemini key needed; uses StubLLM):
python live_portfolio.py --mode dry --provider stub --once

# Live (real money): intentionally hard
ALPACA_ALLOW_LIVE=yes python live_portfolio.py --mode live --once   # + typed confirmation

# Benchmark report (CSV already at results/ai/equity.csv) + chart
python live_portfolio.py report
```

## Forward-test methodology & the benchmark report

From the first run, the harness records start equity and start prices and tracks two
benchmarks every run:

- **Equal-weight universe** buy-and-hold (the honest "did the AI add anything over
  just holding the basket?" test), and
- **SPY** (the market).

`results/ai/equity.csv` logs strategy equity, both benchmark equities, gross exposure,
turnover, and the exposure multiplier each run. `results/ai/decisions.csv` logs every
symbol's score, rationale, target vs current weight, and any order. `live_portfolio.py
report` prints returns vs both benchmarks, strategy max drawdown, and average turnover,
and saves an equity+drawdown chart to `results/ai/report.png`.

**Read it honestly:** a few days of outperformance is noise. You need weeks, and you're
looking for the strategy to beat *both* benchmarks net of costs with tolerable drawdown
and turnover. Underperformance is the likely outcome and is a valid result.

## Inventory integration (hype-at-entry)

When the AI **opens a new position**, the live hype score is captured at that moment and
stamped as `hype_at_entry` in the inventory metadata, along with `entry_date`,
`strategy_tag`, `expected_qty`, and `managed=True`. It is **never backfilled** for
pre-existing positions, and consumers treat a null `hype_at_entry` as **unknown, not 0**.
Positions discovered with no local record default to `managed=False` and are **walled
off** — the AI never trades something it has no record of (the pre-existing AAPL seed is
the one explicit opt-in).

## Conviction gating (the exit logic)

Actions are gated on **conviction = score × confidence**, never raw score, so a weak
or low-confidence read can't trigger a big trade (`portfolio/constructor.py`, band =
`AI_NEUTRAL_BAND`, default 0.10):

- **conviction > +band** → eligible BUY candidate (ranked/weighted by conviction).
- **|conviction| ≤ band** → NEUTRAL: a *held* name is trimmed toward the `max_weight`
  cap (de-concentrate) but **not exited**; an unheld name stays flat.
- **conviction < −band** → decisive **EXIT to 0** (strong, confident negatives leave).

So a 95%-concentrated holding on a −0.10 score @ 0.50 confidence (conviction −0.05,
neutral) becomes **"trim toward the ~20% cap,"** not "sell everything." An optional
per-rebalance **turnover cap** (`AI_TURNOVER_CAP`, off by default) scales all target
moves so the book can't fully flip in one day; caps are then approached over several
rebalances.

## Scheduling (optional — NOT installed)

By default **no scheduler is installed**; trigger dry passes yourself:

```bash
python live_portfolio.py --mode dry --once
```

If you *want* a daily dry run, `scripts/daily_dry_run.sh` (compute + log + audit, no
orders) plus a ready-to-load macOS `launchd` plist
(`scripts/com.tradingbot.dailydryrun.plist`, with install/uninstall commands and a
cron alternative inside it) are provided for you to install **manually**. Nothing
schedules itself.

## Rate limits (free Gemini tier)

The free tier limits **requests per minute** (and per day). One rebalance = ~15 calls;
a single daily run is well within limits, but running several full passes back-to-back
will hit **HTTP 429 RESOURCE_EXHAUSTED**. When that happens the guardrails kick in —
each failed call defaults to **score 0 / no change** (logged in the audit file), so a
rate-limited run simply produces a neutral book rather than crashing or trading wildly.
Mitigations: `AI_LLM_SLEEP` spaces out per-symbol calls; 429s back off ~20s before
retry. Don't judge day-to-day score stability from rapid manual passes — use the daily
schedule, whose once-a-day cadence stays under the quota.

## Costs

Alpaca US equities are commission-free; market orders still pay the spread/slippage, and
intraday market fills differ from any "close" assumption. Turnover is logged so cost drag
is visible. Schedule `--once` at a consistent time each trading day.

## Honest caveats

- **No historical validation** of the news signal — this is forward-only.
- **Small universe** (~15 names) ⇒ concentrated, noisy results; one name moves the book.
- **Crowded space** — news/LLM trading is widely attempted; edges decay.
- **LLM risk** — models hallucinate, are inconsistent run-to-run, and can be swayed by
  PR-speak; expect to iterate on the prompt. The audit log exists for exactly this.
- **News ≠ alpha** — a spike can be routine (scheduled earnings), and sentiment is
  double-edged (momentum vs. overextension). Measuring/scoring it is separate from
  whether it predicts returns.
- **Survivorship-free, but live-data-limited** — free Alpaca news is Benzinga via IEX;
  coverage gaps exist. Treat early results as directional, not conclusive.
```
