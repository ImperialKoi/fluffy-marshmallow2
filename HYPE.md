# Hype Tracker (`signals/hype.py`)

Continuously measure which stocks have **unusual attention** ("hype"), as a SIGNAL
the Phase 3 AI will later consume. This module **measures only** — it does not trade,
and it is not wired into order placement.

> Shared INPUT for the Phase 3 AI. Measuring hype is deliberately separate from
> deciding how to use it.

## Score

`score(symbol)` returns a normalized hype score in **[0, 1]** plus the per-component
breakdown, the components actually used, and which were missing:

```python
from signals.hype import HypeTracker
HypeTracker().score("AAPL")
# {'symbol':'AAPL', 'score':0.61,
#  'components':{'news_velocity':0.82,'rel_volume':0.55,'price_move':0.47},
#  'used':[...], 'missing':['google_trends','social'], 'raw':{...}, 'ts':'...'}
```

The score is the **weighted mean of the available components only** (weights from
`config.HYPE_WEIGHTS` renormalize over what succeeded). If nothing is available the
score is `NaN` (never an exception).

## Components

### Always-on (free, reuse existing infra)

| component | what it measures | how |
|---|---|---|
| `news_velocity` | article flow vs normal | count in last 24h ÷ trailing baseline daily rate (from the `get_info` news tool), saturated to [0,1] |
| `rel_volume` | today's volume vs normal | `today_volume / trailing-N-day average`, saturated to [0,1] |
| `price_move` | unusual price move | z-score of the latest return vs a trailing window → `logistic(|z|−2)` |

Saturation maps a ratio `r` to `r/(r+1)` (1→0.5, 2→0.67, 3→0.75): unbounded
attention compresses into [0,1) without a hard cap dominating the blend.

### Optional (OFF by default, behind config flags)

Enabling these **reverses the project's earlier "no social sources" decision**, so
they are strictly opt-in and need extra deps/keys:

| component | flag | needs |
|---|---|---|
| `google_trends` | `HYPE_ENABLE_GOOGLE_TRENDS` | `pytrends` (free, flaky/rate-limited) |
| `social` (Reddit) | `HYPE_ENABLE_REDDIT` | `praw` + `REDDIT_CLIENT_ID/SECRET/USER_AGENT` |
| `social` (StockTwits) | `HYPE_ENABLE_STOCKTWITS` | public HTTP API (rate-limited), no extra package |

## Rank & snapshot

```python
t = HypeTracker()
t.rank(["AAPL","MSFT","TSLA"])   # most-hyped first (NaN scores last)
t.snapshot()                      # append timestamped scores to config.HYPE_HISTORY_CSV
```

`rank(watchlist)` defaults to `config.HYPE_WATCHLIST`; pass `config.HYPE_DISCOVERY`
(a broader list) to surface hyped names you don't hold yet. `snapshot()` writes a
timestamped row per symbol so hype is a tracked **time series**, not a momentary read.

## Robustness

Every component runs in isolation: a missing or failing source (network down, no
data, optional dep absent) is caught, recorded in `missing`, and excluded from the
weighted average — the score never crashes. The result always reports exactly which
components contributed.

## Configuration (`config.py`)

`HYPE_WATCHLIST`, `HYPE_DISCOVERY`, `HYPE_RVOL_WINDOW` (20), `HYPE_PRICE_WINDOW`
(20), `HYPE_NEWS_BASELINE_DAYS` (7), `HYPE_WEIGHTS`, the three
`HYPE_ENABLE_*` flags (all `False`), and history paths `HYPE_DB` / `HYPE_HISTORY_CSV`.

Data providers are injectable (`news_fn`, `bars_fn`) for testing and for wiring a
live price feed; defaults use the `get_info` news tool and Alpaca bars (falling back
to the bundled cache offline).

## CLI

```bash
python show_state.py hype                                  # rank the config watchlist
python show_state.py hype --watchlist AAPL,MSFT,NVDA       # ad-hoc list
python show_state.py hype --discovery --snapshot          # broad list + log to history
```

## Caveats (read before trusting this)

- **Hype is noisy and gameable.** Mentions and volume can be manufactured.
- **A spike may be routine, not mania** — scheduled earnings, index rebalances, and
  dividends all generate news/volume without "hype."
- **Free social/Trends data is messy** and rate-limited; treat optional components as
  best-effort, and they are off by default for a reason.
- **Hype is double-edged**: high attention can mean momentum *or* overextension/
  exhaustion. This module only *measures* it; the AI decides direction.
- The offline price fallback uses the **bundled 2013–2018 cache**; for current
  relative-volume/price-move set Alpaca keys so `bars_fn` pulls recent bars.

## How the Phase 3 AI will consume this

The AI will read `score(symbol)` as a context feature (with the component breakdown,
not just the scalar) and `rank(discovery_list)` to surface candidates getting unusual
attention. Hype-at-entry can be stamped into the inventory metadata
([INVENTORY.md](INVENTORY.md)) so the AI can later compare attention at entry vs now.
Because hype is double-edged, the AI must combine it with price/structure — it is an
input, not a trade trigger. **Not wired into trading in this phase.**
