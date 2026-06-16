"""
RSS news adapter (free, keyless).

Pulls per-ticker headlines from public RSS feeds and normalizes them. These are
structured feeds (RSS 2.0 / Atom), NOT HTML scraping, so items are not flagged as
scraped. Feeds used (each is best-effort; a dead feed is skipped):

  * Yahoo Finance : https://feeds.finance.yahoo.com/rss/2.0/headline?s=<T>   (last ~25 items)
  * Google News   : https://news.google.com/rss/search?q=<T>+stock          (keyless query feed)
  * Nasdaq        : https://www.nasdaq.com/feed/rssoutbound?symbol=<T>

Licensing/limits: these are public RSS endpoints intended for aggregators; there
is no published hard rate limit, but we cache and back off to be polite. Respect
each provider's terms for redistribution. Parsing uses `feedparser` if installed,
otherwise a small stdlib ElementTree fallback.
"""

from __future__ import annotations

import logging
from datetime import timezone

from .base import Source
from .. import http
from ..schema import Item, NEWS, make_id, to_utc

log = logging.getLogger("info_tool.rss_news")

# {t} is replaced with the (url-safe) ticker.
FEEDS = {
    "yahoo": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US",
    "google": "https://news.google.com/rss/search?q={t}+stock&hl=en-US&gl=US&ceid=US:en",
    "nasdaq": "https://www.nasdaq.com/feed/rssoutbound?symbol={t}",
}

# A browser-ish UA avoids occasional 403s from these endpoints.
_UA = ("Mozilla/5.0 (compatible; trading-bot-info/1.0; +https://example.com/bot)")

# Per-feed HTTP timeout (seconds). RSS feeds are best-effort; keep it short so a
# slow/dead feed fails fast instead of stalling the whole request.
_RSS_TIMEOUT = 6


class RSSNews(Source):
    name = "rss_news"
    item_types = (NEWS,)
    default_on = True

    def __init__(self, feeds: dict | None = None):
        self.feeds = feeds or FEEDS

    def fetch(self, ticker: str, limit: int = 50) -> list[Item]:
        items: list[Item] = []
        per_feed = max(5, int(limit))
        for feed_name, template in self.feeds.items():
            url = template.format(t=ticker.upper())
            try:
                # Fail fast: per-ticker RSS feeds are best-effort and one flaky feed
                # must not dominate latency. Short timeout, single attempt.
                text = http.get_text(url, headers={"User-Agent": _UA},
                                     timeout=_RSS_TIMEOUT, retries=1)
            except Exception as e:  # noqa: BLE001 — one feed failing must not kill the rest
                log.warning("RSS feed '%s' failed for %s: %s", feed_name, ticker, e)
                continue
            try:
                entries = _parse_feed(text)
            except Exception as e:  # noqa: BLE001
                log.warning("RSS parse '%s' failed for %s: %s", feed_name, ticker, e)
                continue
            for e in entries[:per_feed]:
                link = e.get("link", "")
                title = e.get("title", "")
                if not (link or title):
                    continue
                items.append(Item(
                    id=make_id(self.name, native_id=e.get("guid"), url=link, headline=title),
                    tickers=[ticker.upper()],
                    published_utc=to_utc(e.get("published")),
                    source=f"rss/{feed_name}",
                    item_type=NEWS,
                    headline=title,
                    summary=e.get("summary", ""),
                    url=link,
                    extra={"feed": feed_name},
                ))
        return items


# --------------------------------------------------------------------------- #
# parsing: prefer feedparser, fall back to stdlib ElementTree
# --------------------------------------------------------------------------- #
def _parse_feed(text: str) -> list[dict]:
    try:
        import feedparser
    except ImportError:
        return _parse_feed_stdlib(text)
    parsed = feedparser.parse(text)
    out = []
    for e in parsed.entries:
        published = None
        if getattr(e, "published_parsed", None):
            import calendar
            published = calendar.timegm(e.published_parsed)  # epoch (UTC)
        out.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "summary": getattr(e, "summary", ""),
            "guid": getattr(e, "id", None),
            "published": published if published is not None else getattr(e, "published", None),
        })
    return out


def _parse_feed_stdlib(text: str) -> list[dict]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(text)
    out = []
    # RSS 2.0: channel/item
    for item in root.iter("item"):
        out.append({
            "title": _findtext(item, "title"),
            "link": _findtext(item, "link"),
            "summary": _findtext(item, "description"),
            "guid": _findtext(item, "guid"),
            "published": _findtext(item, "pubDate"),
        })
    if out:
        return out
    # Atom fallback: entry/link[@href]
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        link_el = entry.find("a:link", ns)
        out.append({
            "title": _findtext(entry, "{http://www.w3.org/2005/Atom}title"),
            "link": link_el.get("href") if link_el is not None else "",
            "summary": _findtext(entry, "{http://www.w3.org/2005/Atom}summary"),
            "guid": _findtext(entry, "{http://www.w3.org/2005/Atom}id"),
            "published": _findtext(entry, "{http://www.w3.org/2005/Atom}updated"),
        })
    return out


def _findtext(el, tag) -> str:
    child = el.find(tag)
    return (child.text or "").strip() if child is not None and child.text else ""
