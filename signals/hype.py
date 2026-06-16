"""
Hype tracker — measure unusual attention on a stock. SIGNAL ONLY; it does not trade.

This module quantifies "hype" from free sources the project already has, and is a
shared input the Phase 3 AI will later consume. It MEASURES; deciding how to act on
hype (momentum vs. overextension) is a separate, later concern.

Always-on components (free, reuse existing infra, no new paid/social deps):
  * news_velocity : article count in the last 24h vs a trailing baseline rate,
                    via the get_info news tool (tools/info).
  * rel_volume    : today's volume vs a trailing average (relative volume).
  * price_move    : magnitude of the latest return as a z-score vs recent history.

Optional components (OFF by default, behind config flags; enabling social/Trends
reverses the project's earlier "no social sources" decision):
  * google_trends : search interest via pytrends (free, flaky/rate-limited).
  * social        : Reddit (PRAW) / StockTwits mention counts.

Robustness: a missing or failing source NEVER crashes the score. Only the
components that succeeded are weighted (weights renormalize), and the result
records exactly which components were used vs missing.

CAVEATS (see HYPE.md): hype is noisy and gameable; a news spike may be routine
(scheduled earnings) rather than mania; free social/Trends data is messy; the score
is a relative attention gauge, not a forecast.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta

import config

log = logging.getLogger("signals.hype")


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _saturate(ratio: float) -> float:
    """Map a non-negative ratio to [0,1): 1->0.5, 2->0.67, 3->0.75, 5->0.83."""
    r = max(0.0, ratio)
    return r / (r + 1.0)


# --------------------------------------------------------------------------- #
# default data providers (injectable for tests / live)
# --------------------------------------------------------------------------- #
def default_news_fn(symbol: str, limit: int = 100):
    """Recent news items via the project's get_info tool (news only)."""
    from tools.info import get_info
    return get_info(symbol, types=["news"], limit=limit, use_cache=True)


def default_bars_fn(symbol: str):
    """Recent daily OHLCV. Tries Alpaca (live), falls back to the bundled cache."""
    # live path
    try:
        import os
        if os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_LIVE_KEY"):
            from live_trader import fetch_recent_bars
            df = fetch_recent_bars(symbol, lookback_days=120)
            if df is not None and len(df):
                return df
    except Exception as e:  # noqa: BLE001
        log.debug("alpaca bars unavailable for %s: %s", symbol, e)
    # offline fallback: bundled cache (last available bars)
    try:
        from data import loader
        return loader.load_csv(config.CACHE_CSV, symbol)
    except Exception as e:  # noqa: BLE001
        log.debug("cache bars unavailable for %s: %s", symbol, e)
        return None


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #
class HypeTracker:
    def __init__(self, news_fn=None, bars_fn=None, weights: dict = None,
                 rvol_window: int = None, price_window: int = None,
                 news_baseline_days: int = None,
                 enable_google_trends: bool = None, enable_reddit: bool = None,
                 enable_stocktwits: bool = None, history_csv: str = None):
        self.news_fn = news_fn or default_news_fn
        self.bars_fn = bars_fn or default_bars_fn
        self.weights = dict(weights or config.HYPE_WEIGHTS)
        self.rvol_window = rvol_window or config.HYPE_RVOL_WINDOW
        self.price_window = price_window or config.HYPE_PRICE_WINDOW
        self.news_baseline_days = news_baseline_days or config.HYPE_NEWS_BASELINE_DAYS
        self.enable_google_trends = (config.HYPE_ENABLE_GOOGLE_TRENDS
                                     if enable_google_trends is None else enable_google_trends)
        self.enable_reddit = (config.HYPE_ENABLE_REDDIT
                              if enable_reddit is None else enable_reddit)
        self.enable_stocktwits = (config.HYPE_ENABLE_STOCKTWITS
                                  if enable_stocktwits is None else enable_stocktwits)
        self.history_csv = history_csv or config.HYPE_HISTORY_CSV

    # -- public API -------------------------------------------------------- #
    def score(self, symbol: str, now: datetime = None) -> dict:
        """Normalized hype score in [0,1] plus the component breakdown.

        Each component is computed defensively; failures/unavailable sources are
        recorded in `missing` and excluded from the weighted average."""
        symbol = symbol.upper()
        now = now or datetime.now(timezone.utc)
        components, raw, missing = {}, {}, []

        for name, fn in (
            ("news_velocity", lambda: self._news_velocity(symbol, now)),
            ("rel_volume", lambda: self._rel_volume(symbol)),
            ("price_move", lambda: self._price_move(symbol)),
            ("google_trends", lambda: self._google_trends(symbol)),
            ("social", lambda: self._social(symbol)),
        ):
            try:
                comp, detail = fn()
            except Exception as e:  # noqa: BLE001 — one bad source must not crash the score
                log.warning("hype component '%s' failed for %s: %s", name, symbol, e)
                comp, detail = None, {"error": str(e)}
            if comp is None:
                missing.append(name)
                if detail:
                    raw[name] = detail
            else:
                components[name] = float(max(0.0, min(1.0, comp)))
                raw[name] = detail

        used = list(components)
        if used:
            wsum = sum(self.weights.get(n, 1.0) for n in used)
            score = sum(self.weights.get(n, 1.0) * components[n] for n in used) / wsum
        else:
            score = float("nan")

        return {"symbol": symbol, "score": score, "components": components,
                "used": used, "missing": missing, "raw": raw,
                "ts": now.isoformat()}

    def rank(self, watchlist=None, now: datetime = None) -> list[dict]:
        """Watchlist sorted by hype, most-hyped first (NaN scores sort last)."""
        watchlist = watchlist or config.HYPE_WATCHLIST
        scored = [self.score(s, now=now) for s in watchlist]
        scored.sort(key=lambda r: (r["score"] if r["score"] == r["score"] else -1.0),
                    reverse=True)
        return scored

    def snapshot(self, watchlist=None, now: datetime = None) -> list[dict]:
        """Append timestamped scores to the history log so hype is a time series."""
        import csv
        import os

        rows = self.rank(watchlist, now=now)
        os.makedirs(os.path.dirname(self.history_csv) or ".", exist_ok=True)
        new = not os.path.exists(self.history_csv)
        cols = ["ts", "symbol", "score", "news_velocity", "rel_volume", "price_move",
                "google_trends", "social", "used", "missing"]
        with open(self.history_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            for r in rows:
                c = r["components"]
                w.writerow({
                    "ts": r["ts"], "symbol": r["symbol"],
                    "score": _fmt(r["score"]),
                    "news_velocity": _fmt(c.get("news_velocity")),
                    "rel_volume": _fmt(c.get("rel_volume")),
                    "price_move": _fmt(c.get("price_move")),
                    "google_trends": _fmt(c.get("google_trends")),
                    "social": _fmt(c.get("social")),
                    "used": "|".join(r["used"]), "missing": "|".join(r["missing"]),
                })
        return rows

    # -- always-on components --------------------------------------------- #
    def _news_velocity(self, symbol, now):
        items = self.news_fn(symbol) or []
        times = []
        for it in items:
            ts = getattr(it, "published_utc", None)
            if ts is None and isinstance(it, dict):
                ts = it.get("published_utc")
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                times.append(ts)
        if not times:
            return None, {"reason": "no dated news"}

        n24 = sum(1 for t in times if t >= now - timedelta(hours=24))
        window_start = now - timedelta(days=self.news_baseline_days)
        in_window = [t for t in times if t >= window_start]
        # baseline daily rate over the trailing window
        daily_rate = max(len(in_window) / self.news_baseline_days, 1e-9)
        ratio = n24 / daily_rate
        comp = _saturate(ratio)
        return comp, {"n_24h": n24, "n_baseline_window": len(in_window),
                      "baseline_daily_rate": round(daily_rate, 3),
                      "velocity_ratio": round(ratio, 3),
                      "total_items": len(times)}

    def _rel_volume(self, symbol):
        df = self.bars_fn(symbol)
        if df is None or "volume" not in getattr(df, "columns", []) or len(df) < self.rvol_window + 1:
            return None, {"reason": "insufficient volume data"}
        vol = df["volume"].astype(float)
        today = float(vol.iloc[-1])
        base = float(vol.iloc[-(self.rvol_window + 1):-1].mean())
        if base <= 0:
            return None, {"reason": "zero baseline volume"}
        rvol = today / base
        return _saturate(rvol), {"rvol": round(rvol, 3), "today_volume": today,
                                 "baseline_avg_volume": round(base, 1)}

    def _price_move(self, symbol):
        df = self.bars_fn(symbol)
        if df is None or "close" not in getattr(df, "columns", []) or len(df) < self.price_window + 2:
            return None, {"reason": "insufficient price data"}
        rets = df["close"].astype(float).pct_change().dropna()
        if len(rets) < self.price_window + 1:
            return None, {"reason": "insufficient returns"}
        recent = float(rets.iloc[-1])
        base = rets.iloc[-(self.price_window + 1):-1]
        sd = float(base.std())
        if sd <= 0:
            return None, {"reason": "zero volatility baseline"}
        z = (recent - float(base.mean())) / sd
        comp = _logistic(abs(z) - 2.0)   # |z|=2 -> 0.5, larger moves -> higher
        return comp, {"return": round(recent, 4), "z_score": round(z, 2),
                      "direction": "up" if recent >= 0 else "down"}

    # -- optional components (OFF by default) ----------------------------- #
    def _google_trends(self, symbol):
        if not self.enable_google_trends:
            return None, None
        try:
            from pytrends.request import TrendReq
        except ImportError:
            return None, {"reason": "pytrends not installed"}
        pt = TrendReq(hl="en-US", tz=0)
        pt.build_payload([symbol], timeframe="now 7-d")
        data = pt.interest_over_time()
        if data is None or data.empty or symbol not in data:
            return None, {"reason": "no trends data"}
        series = data[symbol].astype(float)
        recent = float(series.iloc[-1])
        base = float(series.iloc[:-1].mean()) or 1.0
        ratio = recent / base
        return _saturate(ratio), {"recent": recent, "baseline": round(base, 1),
                                  "ratio": round(ratio, 3)}

    def _social(self, symbol):
        if not (self.enable_reddit or self.enable_stocktwits):
            return None, None
        counts = []
        detail = {}
        if self.enable_stocktwits:
            try:
                from tools.info import http
                data = http.get_json(
                    f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
                    headers={"User-Agent": "trading-bot-hype/1.0"})
                n = len(data.get("messages", []))
                counts.append(n)
                detail["stocktwits_messages"] = n
            except Exception as e:  # noqa: BLE001
                detail["stocktwits_error"] = str(e)
        if self.enable_reddit:
            try:
                detail.update(self._reddit_count(symbol))
                counts.append(detail.get("reddit_mentions", 0))
            except Exception as e:  # noqa: BLE001
                detail["reddit_error"] = str(e)
        if not counts:
            return None, detail or {"reason": "no social data"}
        # crude saturation: ~30 messages -> ~0.5
        total = sum(counts)
        return _saturate(total / 30.0), detail | {"total_mentions": total}

    def _reddit_count(self, symbol):
        import os
        cid = os.environ.get("REDDIT_CLIENT_ID")
        csec = os.environ.get("REDDIT_CLIENT_SECRET")
        ua = os.environ.get("REDDIT_USER_AGENT", "trading-bot-hype/1.0")
        if not (cid and csec):
            return {"reddit_error": "REDDIT_CLIENT_ID/SECRET not set"}
        import praw
        reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
        n = sum(1 for _ in reddit.subreddit("wallstreetbets+stocks+investing")
                .search(symbol, time_filter="day", limit=50))
        return {"reddit_mentions": n}


def _fmt(x):
    return "" if x is None or (isinstance(x, float) and x != x) else round(float(x), 4)
