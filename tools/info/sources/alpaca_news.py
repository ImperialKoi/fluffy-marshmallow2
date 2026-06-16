"""
Alpaca News adapter (PRIMARY news source).

Real Benzinga headlines via alpaca-py's NewsClient, using the Alpaca account the
project already configures (ALPACA_KEY / ALPACA_SECRET). Structured: timestamps,
tickers, headline, summary, url, author.

  * Docs: https://docs.alpaca.markets/docs/historical-news-data
  * Provider: Benzinga (via Alpaca Market Data).
  * Coverage: news back to 2015, ~130+ articles/day.
  * Rate limit: Alpaca Market Data limits apply (200 req/min on the free plan).
  * Cost: free with an Alpaca account (paper keys work for data).

Degrades gracefully: if alpaca-py isn't installed or keys are absent, available()
returns False and the retriever simply skips it.
"""

from __future__ import annotations

import logging
import os

from .base import Source
from ..schema import Item, NEWS, make_id, to_utc

log = logging.getLogger("info_tool.alpaca_news")


class AlpacaNews(Source):
    name = "alpaca_news"
    item_types = (NEWS,)
    default_on = True

    def _keys(self):
        key = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_LIVE_KEY")
        secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_LIVE_SECRET")
        return key, secret

    def available(self) -> bool:
        key, secret = self._keys()
        if not (key and secret):
            return False
        try:
            import alpaca  # noqa: F401
        except ImportError:
            return False
        return True

    def unavailable_reason(self) -> str:
        key, secret = self._keys()
        if not (key and secret):
            return "ALPACA_KEY / ALPACA_SECRET not set"
        return "alpaca-py not installed"

    def fetch(self, ticker: str, limit: int = 50) -> list[Item]:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        key, secret = self._keys()
        client = NewsClient(key, secret)
        req = NewsRequest(symbols=ticker, limit=min(int(limit), 50),
                          include_content=False, exclude_contentless=True)
        resp = client.get_news(req)

        # NewsClient returns a NewsSet whose .data maps "news" -> [News, ...];
        # be defensive about shape across alpaca-py versions.
        raw = getattr(resp, "data", resp)
        articles = raw.get("news", []) if isinstance(raw, dict) else (raw or [])

        items: list[Item] = []
        for n in articles:
            symbols = list(getattr(n, "symbols", None) or [ticker])
            items.append(Item(
                id=make_id(self.name, native_id=getattr(n, "id", None)),
                tickers=[s.upper() for s in symbols],
                published_utc=to_utc(getattr(n, "created_at", None)),
                source="alpaca/benzinga",
                item_type=NEWS,
                headline=getattr(n, "headline", "") or "",
                summary=getattr(n, "summary", "") or "",
                url=getattr(n, "url", "") or "",
                extra={
                    "author": getattr(n, "author", None),
                    "native_source": getattr(n, "source", None),
                    "updated_utc": _iso(to_utc(getattr(n, "updated_at", None))),
                },
            ))
        return items


def _iso(dt):
    return dt.isoformat() if dt else None
