"""
Central configuration for the backtest system.

Everything that you might want to tweak between runs lives here so the rest of
the code stays clean. These are *backtest* settings; when you eventually move to
paper/live trading the same numbers (costs, sizing, stops) should carry over so
that live behaviour matches what you tested.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Where the cached real-market dataset lives (S&P 500, 2013-2018, daily bars).
CACHE_CSV = "data_cache/sp500_5yr.csv"

# Default symbol to trade in the demo. Any S&P 500 ticker in the dataset works
# (AAPL, MSFT, AMZN, JPM, KO, XOM, ...).
SYMBOL = "AAPL"

# Optional date filter (None = use everything available for the symbol).
START = None          # e.g. "2014-01-01"
END = None            # e.g. "2017-12-31"

# ---------------------------------------------------------------------------
# Account / portfolio
# ---------------------------------------------------------------------------
INITIAL_CASH = 100_000.0

# ---------------------------------------------------------------------------
# Trading costs (realism — never backtest without these)
# ---------------------------------------------------------------------------
# Commission per trade as a fraction of notional. Alpaca US equities are
# commission-free, so 0.0 is realistic there; bump it up to stress-test.
COMMISSION_BPS = 0.0          # basis points (1 bp = 0.01%)
# Slippage: how much worse than the quoted price you actually fill, in bps.
# Applied against you on both entry and exit.
SLIPPAGE_BPS = 5.0            # 5 bps = 0.05%

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
# Fraction of current equity to deploy when fully invested in one position.
POSITION_FRACTION = 0.95
# Per-trade stop loss / take profit as a fraction of entry price. None disables.
STOP_LOSS_PCT = 0.08          # exit if price falls 8% below entry
TAKE_PROFIT_PCT = None        # e.g. 0.20 to lock in 20% gains
# Portfolio kill switch: if total drawdown from peak exceeds this, stop trading.
MAX_DRAWDOWN_KILL = 0.25      # 25%

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252   # used to annualise return / volatility
RISK_FREE_RATE = 0.0          # annual; keep 0 for simplicity

# ---------------------------------------------------------------------------
# Portfolio inventory (portfolio/inventory.py)
# ---------------------------------------------------------------------------
# Local store for metadata Alpaca doesn't keep (entry date, rationale tag, target
# weight, stop level, hype-at-entry, expected qty for reconciliation) and for the
# timestamped snapshot history. Alpaca remains the source of truth for quantities
# and cost basis; these files never override it.
PORTFOLIO_DB = "results/portfolio/inventory.db"        # SQLite: metadata + snapshots
PORTFOLIO_HISTORY_CSV = "results/portfolio/history.csv"  # flat snapshot log (forward-test record)

# ---------------------------------------------------------------------------
# Hype tracker (signals/hype.py)
# ---------------------------------------------------------------------------
# Watchlist to score by default (a sector-spread of liquid names present in the
# bundled dataset). DISCOVERY is a broader list to surface hyped names not held.
HYPE_WATCHLIST = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM", "JNJ", "XOM",
]
HYPE_DISCOVERY = HYPE_WATCHLIST  # override with a larger universe for discovery

# Baseline / lookback windows for the always-on components.
HYPE_RVOL_WINDOW = 20          # trailing days for relative-volume baseline
HYPE_PRICE_WINDOW = 20         # trailing days for the return z-score baseline
HYPE_NEWS_BASELINE_DAYS = 7    # window over which to estimate a baseline news rate

# Component weights (only the AVAILABLE components are used; weights renormalize).
HYPE_WEIGHTS = {
    "news_velocity": 1.0,
    "rel_volume": 1.0,
    "price_move": 1.0,
    "google_trends": 1.0,   # only if enabled below
    "social": 1.0,          # only if enabled below
}

# Optional sources — OFF by default. Enabling social/Trends REVERSES the project's
# earlier "no social sources" decision, so it is strictly opt-in (and needs deps/keys).
HYPE_ENABLE_GOOGLE_TRENDS = False   # pytrends (free, flaky/rate-limited)
HYPE_ENABLE_REDDIT = False          # PRAW (needs REDDIT_CLIENT_ID/SECRET/USER_AGENT)
HYPE_ENABLE_STOCKTWITS = False      # StockTwits public API (rate-limited)

# History log for hype scores (so hype is a tracked time series, not a momentary read).
HYPE_DB = "results/hype/hype.db"
HYPE_HISTORY_CSV = "results/hype/history.csv"

# ---------------------------------------------------------------------------
# Phase 3 — AI news-driven cross-sectional strategy (agents/, live_portfolio.py)
# FORWARD-TEST ONLY: runs live on paper against real-time news; no historical
# backtest of the news signal exists. This is an experiment, not a proven edge.
# ---------------------------------------------------------------------------
# Universe: liquid US large caps tradable on Alpaca (>= 10; default ~15).
AI_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM", "JNJ",
    "XOM", "UNH", "V", "PG", "HD", "KO",
]

# LLM provider/model (provider behind a thin wrapper so it can be swapped).
AI_LLM_PROVIDER = "gemini"
AI_MODEL = "gemini-3.5-flash"     # verify availability for your SDK/key
# LLM fallback chain when Gemini is unreachable/overloaded:
#   1st Gemini  ->  2nd Cohere (COHERE_API_KEY)  ->  3rd OpenAI/ChatGPT (OPENAI_API_KEY)
# Each tier is included only if its key is present; the chain rolls over on a 503/
# timeout/rate-limit ("unable to reach") error and stops on a usable answer.
AI_COHERE_MODEL = "command-a-03-2025"   # 2nd choice (current Cohere model; older
                                        # 'command-r'/'command-r-plus' were retired 2025).
                                        # If your account 404s, set to one listed at
                                        # https://docs.cohere.com/docs/models
AI_OPENAI_MODEL = "gpt-4o-mini"   # 3rd choice

# COMPLETE FREEDOM: when True the AI may trade/sell/protect ANY position it holds
# (the `managed` wall-off is bypassed) AND buy tickers beyond the base universe via an
# LLM discovery step. Deterministic risk sizing (max weight/gross, turnover cap, kill
# switch, degraded guard, protective stops) still applies. Long/flat only (no shorting).
AI_FREE_TRADE = True
AI_DISCOVERY_COUNT = 10           # max new tickers the AI may propose per rebalance (0=off)
AI_LLM_TIMEOUT = 30               # seconds per call
AI_LLM_RETRIES = 3
# Seconds to sleep between per-symbol LLM calls. Free Gemini tiers rate-limit by
# requests-per-minute; a small spacing avoids 429s on a 15-call rebalance. A
# once-daily run is well within free limits, so default 0; raise if you see 429s.
AI_LLM_SLEEP = 0.0
AI_NEWS_LIMIT = 12                # max info items per symbol fed to the LLM
AI_EXPOSURE_PASS = False          # optional portfolio-level risk-on/off LLM pass
# Batched scoring: score the WHOLE basket in ONE LLM call (a JSON array) instead of
# one call per symbol. Cuts API calls ~15x -> fits tight free tiers and is far cheaper/
# faster. Symbols are chunked so the prompt stays bounded.
AI_BATCH_SCORING = True
AI_BATCH_ITEMS_PER_SYMBOL = 6     # info items per symbol in a batch prompt (token budget)
AI_BATCH_CHUNK = 25               # max symbols per LLM call (split into chunks if more)
# Safety: if fewer than this fraction of symbols return a USABLE score (e.g. during
# a Gemini 503/429 outage), SKIP the rebalance and hold current positions rather than
# trading on near-zero information.
AI_MIN_OK_FRACTION = 0.5

# Deterministic portfolio constructor + risk limits (the LLM never sizes trades).
AI_MAX_WEIGHT = 0.20              # max fraction of equity per name
AI_MAX_GROSS = 0.95              # max total invested (rest is cash)
AI_MIN_CASH = 0.05              # minimum cash buffer
AI_MIN_POSITIONS = 3             # diversification floor (scales exposure if fewer)
AI_MAX_POSITIONS = 10            # cap on number of holdings
AI_MIN_SCORE = 0.05             # secondary raw-score floor for a long candidate
AI_WEIGHTING = "confidence"      # "equal" or "confidence" (score x confidence)
AI_DRAWDOWN_KILL = MAX_DRAWDOWN_KILL  # reuse the project kill-switch threshold

# Conviction gating (so a weak signal can't trigger a full liquidation). Actions are
# gated on conviction = score x confidence, NOT raw score:
#   * conviction >  +AI_NEUTRAL_BAND  -> eligible BUY candidate
#   * |conviction| <= AI_NEUTRAL_BAND -> NEUTRAL: a held name is trimmed toward the
#     max-weight cap (de-concentrate), NOT exited; an unheld name stays flat
#   * conviction <  -AI_NEUTRAL_BAND  -> decisive EXIT to 0
AI_NEUTRAL_BAND = 0.10
# Per-rebalance turnover cap (sum of |target_w - current_w|). None = off.
# Default 0.20: target moves are scaled so at most ~20% of the book changes per
# rebalance, so a single day's read can't fully flip the portfolio (and the AAPL
# seed de-concentrates gradually over several rebalances rather than in one shot).
AI_TURNOVER_CAP = 0.20

# Score-stability gate (agents/stability.py): look at the last N runs' per-symbol
# scores in results/ai/decisions.csv and flag names whose sign keeps flipping, so
# the go/no-go is mechanical rather than by eye. Advisory (logged), not a hard block.
AI_STABILITY_RUNS = 5            # how many recent runs to consider
AI_STABILITY_FLIP_THRESHOLD = 2  # >= this many sign flips in the window -> "unstable"
AI_STABILITY_MIN_RUNS = 3        # fewer than this -> "insufficient history"

# Scheduling / state / audit paths.
AI_REBALANCE_SECONDS = 86_400    # for --loop (once per day)
AI_AUDIT_DIR = "results/ai_audit"            # dated JSONL: prompt + raw + parsed per call
AI_STATE_FILE = "results/ai/portfolio_state.json"     # kill-switch high-water mark
AI_DECISIONS_LOG = "results/ai/decisions.csv"         # per-run scores/targets/orders
AI_EQUITY_LOG = "results/ai/equity.csv"               # strategy vs benchmark equity
AI_BENCH_STATE = "results/ai/benchmark_state.json"    # benchmark start prices/equity
AI_BENCHMARK_SPY = "SPY"

# ---------------------------------------------------------------------------
# Always-on service (live_service.py) — two-cadence streaming runner
# Long-running process: streams live bars, runs a cheap deterministic scan every
# ~minute, runs the AI news rebalance every ~hour, and maintains server-side
# protective orders. Market-hours gated. This is NOT a system scheduler — you
# start it yourself; it does not autostart.
# ---------------------------------------------------------------------------
SERVICE_SCAN_INTERVAL_SEC = 60        # fast deterministic scan cadence
SERVICE_REBALANCE_INTERVAL_MIN = 60   # slow AI rebalance cadence
SERVICE_INVENTORY_SYNC_MIN = 15       # periodic Alpaca-authoritative inventory sync
SERVICE_EXTENDED_HOURS = False        # also operate in pre/post-market windows
SERVICE_BUFFER_BARS = 240             # rolling minute bars kept per symbol (~4h; small mem)
SERVICE_WARMUP_BARS = 120             # REST-prefill this many recent minute bars at startup
                                      # so the fast scan has data immediately (no ~30min warm-up)
SERVICE_FEED = "iex"                  # free Alpaca data feed
SERVICE_FAST_STRATEGY = "supertrend"  # deterministic fast-scan strategy (reuses registry)
SERVICE_FAST_MIN_BARS = 30            # min buffered bars before scanning a symbol

# Protective resting orders (server-side, GTC — fire at the exchange even if the
# bot/instance is down). Maintained for MANAGED long positions only. Default is an
# OCO bracket so EACH position rests with BOTH a take-profit (ceiling) and a
# stop-loss (floor) at the exchange.
PROTECT_ENABLED = True
PROTECT_STOP_PCT = 0.08                    # stop-loss (floor) distance below entry
PROTECT_TRAILING_PCT = None               # e.g. 0.05 trailing stop; None = off
PROTECT_TAKE_PROFIT_PCT = 0.20            # take-profit (ceiling) distance above entry
PROTECT_BRACKET_OCO = True                # place an OCO (take-profit limit + stop-loss)

# ---------------------------------------------------------------------------
# Deterministic exit engine (service/risk_exits.py) — runs every fast tick, NO LLM.
# Each managed long is exited the instant it breaches its risk frame, regardless of
# whether any LLM is reachable. This is the "sell even if the AI is offline" guarantee.
# ---------------------------------------------------------------------------
SERVICE_DETERMINISTIC_EXITS = True
RISK_STOP_PCT = PROTECT_STOP_PCT          # floor: stop loss at -8% from entry
RISK_TAKE_PROFIT_PCT = PROTECT_TAKE_PROFIT_PCT  # ceiling: take profit at +20% from entry
RISK_CRASH_PCT = 0.08                     # trailing: sell if it drops 8% from a recent high
RISK_CRASH_LOOKBACK = 30                  # bars (minutes) for the recent-high window
RISK_USE_SR = True                        # also use support/resistance from the live buffer
RISK_SUPPORT_BREAK_BUFFER = 0.0           # sell if price breaks below nearest support by this frac
RISK_CEILING_BUFFER = 0.0                 # sell within this frac of nearest resistance (in profit)

# ---------------------------------------------------------------------------
# Dynamic universe DISCOVERY (universe/) — Phase 3.5, FORWARD-TEST / PAPER ONLY.
# Once per day, a DETERMINISTIC screener over real Alpaca market data proposes new
# high-risk/high-reward candidates (penny / high-vol / high-relative-volume + low
# float), the existing LLM scorer EVALUATES them (it never invents tickers), a
# deterministic gate re-confirms tradability + a hard liquidity floor and has final
# say, then the persisted universe expands and the existing AI rebalance allocates
# within it. Set UNIVERSE_DISCOVERY_ENABLED=False to pin the universe to AI_UNIVERSE.
#
# CAVEATS (see UNIVERSE.md): penny / low-float names are easily manipulated and gap
# hard; market-order fills are worse on thin names; yfinance float data is best-effort
# and often missing. That is exactly why every candidate must clear a liquidity floor,
# why the speculative sleeve is hard-capped, and why these names get TIGHT stops/TPs.
# ---------------------------------------------------------------------------
UNIVERSE_DISCOVERY_ENABLED = True
UNIVERSE_STORE_FILE = "results/universe/universe.json"   # persisted dynamic universe
UNIVERSE_DISCOVERY_LOG = "results/universe/discovery.csv"  # every discovery decision
UNIVERSE_MAX_SIZE = 40              # cap total universe (pinned core + dynamic) so hourly
                                    # scoring stays within free LLM limits; evict on overflow
UNIVERSE_DISCOVERY_INTERVAL_HOURS = 24   # daily cadence (separate from scan/rebalance)

# Deterministic screener thresholds (universe/screener.py) over recent daily bars.
SCREEN_PRICE_CAP = 5.0              # penny / high-risk: only names priced <= this
SCREEN_MIN_DOLLAR_VOLUME = 1_000_000.0   # HARD liquidity floor: min avg daily $ volume so
                                         # the bot can enter AND exit (applies even to pennies)
SCREEN_MIN_VOLATILITY = 0.04       # min realized daily vol (stdev of daily returns) — high risk
SCREEN_RVOL_MIN = 1.5              # min relative volume (today vs trailing avg) — "high volume"
SCREEN_MAX_FLOAT = 75_000_000      # low-float ceiling (shares); best-effort via yfinance
SCREEN_REQUIRE_FLOAT = False       # if True, drop names with no float data (default: keep, fail soft)
SCREEN_LOOKBACK_DAYS = 30          # trailing window for vol / dollar-volume / rel-volume
SCREEN_DAILY_CANDIDATES = 20       # cap the ranked candidate list fed to the LLM (free-tier safe)
SCREEN_MAX_ASSETS_SCANNED = 3000   # cap tradable assets pulled for bars (0 = hype prelist only)
SCREEN_BARS_CHUNK = 200            # symbols per multi-symbol Alpaca bar request
SCREEN_INCLUDE_HYPE = True         # also fold in the hype tracker's most-hyped names
SCREEN_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")  # supported exchanges only

# Discovery LLM evaluation (reuses the news scorer; advisory, never sizes/invents).
DISCOVERY_MIN_CONVICTION = 0.10    # min LLM score*confidence for a candidate to be admitted
DISCOVERY_LLM_SLEEP = AI_LLM_SLEEP # spacing between discovery LLM calls (free-tier limits)

# Speculative sleeve — EXTENDS portfolio/risk.py, never weakens it. A hard cap on total
# equity across ALL speculative/penny names combined, plus a tighter per-name cap, so a
# blowup in thin names can't sink the book. Core (pinned) names keep the normal limits.
SPEC_SLEEVE_ENABLED = True
SPEC_SLEEVE_PCT = 0.15             # max combined equity weight across all speculative names
SPEC_MAX_WEIGHT = 0.05             # tighter per-name cap for a speculative name (vs AI_MAX_WEIGHT)
# TIGHT stops/take-profits for penny/speculative names (vs the core 8%/20%). These feed
# BOTH the server-side protective/OCO orders and the deterministic exit engine, per tier.
SPEC_STOP_PCT = 0.04               # tight stop-loss for speculative names
SPEC_TAKE_PROFIT_PCT = 0.08        # tight take-profit for speculative names
