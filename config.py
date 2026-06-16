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
AI_LLM_TIMEOUT = 30               # seconds per call
AI_LLM_RETRIES = 3
# Seconds to sleep between per-symbol LLM calls. Free Gemini tiers rate-limit by
# requests-per-minute; a small spacing avoids 429s on a 15-call rebalance. A
# once-daily run is well within free limits, so default 0; raise if you see 429s.
AI_LLM_SLEEP = 0.0
AI_NEWS_LIMIT = 12                # max info items per symbol fed to the LLM
AI_EXPOSURE_PASS = False          # optional portfolio-level risk-on/off LLM pass

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
# Optional per-rebalance turnover cap (sum of |target_w - current_w|). None = off.
# When set, target moves are scaled so the book can't fully flip on one day's read;
# caps are then approached over several rebalances rather than instantly.
AI_TURNOVER_CAP = None

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
SERVICE_FEED = "iex"                  # free Alpaca data feed
SERVICE_FAST_STRATEGY = "supertrend"  # deterministic fast-scan strategy (reuses registry)
SERVICE_FAST_MIN_BARS = 30            # min buffered bars before scanning a symbol

# Protective resting orders (server-side, GTC — fire at the exchange even if the
# bot/instance is down). Maintained for MANAGED long positions only.
PROTECT_ENABLED = True
PROTECT_STOP_PCT = STOP_LOSS_PCT          # stop-loss distance below entry (0.08)
PROTECT_TRAILING_PCT = None               # e.g. 0.05 trailing stop; None = off
PROTECT_TAKE_PROFIT_PCT = TAKE_PROFIT_PCT  # take-profit distance above entry; None = off
PROTECT_BRACKET_OCO = False               # if True and a take-profit is set, place an OCO (TP limit + stop)
