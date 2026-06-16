# Real-Time Information Tool (`tools/info/`)

A resilient, pluggable feed of **recent** market information for a ticker — news,
SEC filings, and analyst ratings/price targets — normalized into one schema and
merged/deduped behind a single call. This is the information layer a future AI
decision agent reasons over. Historical backfill is out of scope; this fetches
what's *current*.

> Design mirrors the project's existing convention (`data/loader.py`): one
> interface, many pluggable adapters, env-var credentials, fail-soft. **Not
> investment advice.**

## Quick start

```python
from tools.info import get_info

items = get_info("AAPL", types=["news", "filing", "analyst"], limit=30)
for it in items:
    print(it.published_utc, it.item_type, it.source, "—", it.headline)
```

```python
# Only filings, force a fresh fetch (bypass cache):
get_info("MSFT", types=["filing"], use_cache=False)

# Restrict to specific sources (incl. the opt-in GDELT source):
get_info("NVDA", sources=["edgar_filings", "gdelt"])

# Backtest-friendly: exclude anything published after a point in time:
get_info("AAPL", as_of="2024-06-01T00:00:00Z")
```

## The interface

```python
get_info(ticker, types=None, limit=50, sources=None, as_of=None,
         use_cache=True, cache=None) -> list[Item]
```

| Arg | Meaning |
|---|---|
| `ticker` | Symbol (case-insensitive). |
| `types` | Subset of `{"news","filing","analyst"}`; `None` = all. |
| `limit` | Max items returned after merge+dedup, newest first. |
| `sources` | Subset of source names; `None` = all default-on sources. |
| `as_of` | `datetime`/ISO string; drop items published **after** this UTC instant. Undated items are kept (can't prove they're future). |
| `use_cache` | If `True`, serve fresh-enough cached results per source; `False` forces a refetch. |
| `cache` | Optional `Cache` instance (defaults to the shared on-disk cache). |

Returns a list of `Item`, **deduplicated** (by canonical URL and fuzzy headline)
and **sorted newest first** (undated items last). A failing source never breaks the
call — it is logged and skipped.

> **Note on `limit` and per-type coverage:** `limit` is a single global cap applied
> after sorting newest-first. Because news is the most frequent/recent type, a small
> global limit can crowd out older filings/analyst items. For guaranteed coverage of
> a specific type, call per-type (`types=["filing"]`) or raise `limit`.

## Schema (`Item`)

```python
Item(id, tickers, published_utc, source, item_type, headline, summary, url, extra)
```

- `item_type` ∈ `{"news", "filing", "analyst"}`.
- `published_utc` is timezone-aware UTC (or `None` for undated snapshots like a
  current analyst price-target consensus).
- `extra` holds type-specific structured fields:
  - **filing:** `{form, cik, accession, filing_date, primary_doc_description, event_items, is_insider}`
  - **analyst:** `{firm, action, from_grade, to_grade, price_target, recommendation_distribution, scraped, caveat}`
  - **news:** `{author, native_source, feed, ...}`
  - `extra["scraped"] = True` flags items obtained by scraping rather than a
    structured feed/API (currently only the yfinance analyst data).

## Sources

One adapter per source (`tools/info/sources/`), each behind the same `Source`
interface. Default-on sources are used when the caller doesn't name `sources`.

| Source name | Type(s) | Default | Key needed | Notes |
|---|---|---|---|---|
| `alpaca_news` | news | ✅ | Alpaca keys (optional) | **Primary news.** Benzinga headlines via `alpaca-py`. Skipped if keys/SDK absent. |
| `rss_news` | news | ✅ | none | Yahoo Finance, Google News, Nasdaq RSS feeds (keyless). |
| `edgar_filings` | filing | ✅ | none (UA required) | Official SEC EDGAR submissions API: 8-K, 10-K, 10-Q, Form 4. |
| `yfinance_analyst` | analyst | ✅ | none | Upgrades/downgrades, price targets, recommendation mix. **Scraped — best-effort.** |
| `gdelt` | news | ❌ (opt-in) | none | GDELT DOC 2.0 global news. Noisy by raw ticker, so off by default. |

### Rate limits & licensing

- **Alpaca News** — free with an Alpaca account (paper keys work for data);
  subject to Alpaca Market Data rate limits (200 req/min on the free plan).
  Data © Benzinga via Alpaca. Docs: <https://docs.alpaca.markets/docs/historical-news-data>
- **SEC EDGAR** — free, public domain, **no key**, but **requires a descriptive
  `User-Agent` with a contact email** or it blocks you. Limit ~10 req/sec/IP; we
  cache and stay well under. Set `SEC_EDGAR_USER_AGENT="Your Name your@email.com"`.
  Ref: <https://www.sec.gov/os/accessing-edgar-data>
- **RSS (Yahoo/Google/Nasdaq)** — public RSS endpoints intended for aggregators;
  no published hard limit, but we cache + back off to be polite. Respect each
  provider's terms before redistributing content.
- **GDELT DOC 2.0** — free, keyless; undocumented soft rate limit (429s happen) —
  we back off and cache. Ref: <https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/>

### ⚠️ yfinance caveat (read this)

The analyst adapter uses **yfinance, which scrapes Yahoo Finance** — it is **not an
official API**. Expect gaps, schema drift, and occasional breakage when Yahoo
changes their site. Every analyst item is flagged `extra["scraped"] = True` with a
`caveat`. Each field (upgrades/downgrades, price targets, recommendations) is
fetched independently so one failing doesn't lose the others, and a total failure
returns `[]` rather than raising. Treat this data as best-effort, not ground truth.

## Resilience

- **Timeouts + exponential backoff** with retry on transient errors (timeouts,
  connection errors, 429, 5xx), centralized in `tools/info/http.py`.
- **Fail-soft:** every source runs isolated; an exception (or missing key) logs a
  warning and is skipped. Within `rss_news`, each individual feed is isolated too.
- **Feeds/APIs first:** we use official APIs and structured RSS/Atom; only the
  yfinance path is scraped (and flagged). No HTML scraping of news pages.

## Caching

A small SQLite cache (`results/info_cache.db` by default) keyed by `(source,
ticker)` with a short TTL (default **600 s**), storing exact UTC fetch times.

```python
from tools.info import default_cache
c = default_cache()
c.clear()                      # wipe everything
c.clear(source="rss_news")     # wipe one source
c.clear(ticker="AAPL")         # wipe one ticker
c.purge_expired()              # drop only stale rows
```

Overrides via env: `INFO_CACHE_PATH`, `INFO_CACHE_TTL`. Bypass reads per call with
`use_cache=False`.

## Configuration (env vars only — never hardcoded)

See `.env.example`. Relevant keys:

```bash
ALPACA_KEY=...            # optional; enables the primary news source
ALPACA_SECRET=...
SEC_EDGAR_USER_AGENT="Your Name your@email.com"   # required for EDGAR
# INFO_CACHE_PATH=results/info_cache.db
# INFO_CACHE_TTL=600
```

## Tests

```bash
python tests/test_info_tool.py          # mocked HTTP, no network: dedup, as_of,
                                         # graceful-failure, type filter, cache, adapters
RUN_LIVE_INFO_TESTS=1 \
SEC_EDGAR_USER_AGENT="You you@email.com" \
python tests/test_info_tool.py TestLiveSmoke   # one gated live fetch for AAPL
```

### Live AAPL sample (all three types)

```
[news]    rss/yahoo   2026-06-15  "MAGS To Rags? Magnificent Seven Stocks Are Market Drags"
          rss/google  2026-06-15  "Best Time to Buy Apple Stock? AAPL Price Analysis ..."
[filing]  sec_edgar   2026-05-29  "AAPL 4 — FORM 4"  (10-Q / 8-K / Form 4 also returned)
[analyst] yfinance    2026-06-09  "AAPL: TD Cowen main — Buy → Buy"  [scraped]
```
