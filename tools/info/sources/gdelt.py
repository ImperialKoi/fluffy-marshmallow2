"""
GDELT DOC 2.0 adapter (optional, free, keyless) — global news mentions.

Queries the public GDELT DOC 2.0 API in ArticleList/JSON mode for recent articles
mentioning the ticker. Useful for broad/global coverage beyond US finance feeds.

  * Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
  * Params  : query, mode=ArticleList, format=json, maxrecords<=250, sort=DateDesc
  * Returns : up to ~250 articles (title, url, domain, seendate, language).
  * Limits  : free; undocumented soft rate limit (429s happen) — we back off + cache.
  * Caveat  : searching by raw ticker is noisy, so this source is OFF by default
              (default_on = False). Enable explicitly via sources=["gdelt", ...].

Ref: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

from __future__ import annotations

import logging

from .base import Source
from .. import http
from ..schema import Item, NEWS, make_id, to_utc

log = logging.getLogger("info_tool.gdelt")

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


class Gdelt(Source):
    name = "gdelt"
    item_types = (NEWS,)
    default_on = False  # noisy by raw ticker; opt-in

    def fetch(self, ticker: str, limit: int = 50) -> list[Item]:
        # Constrain to finance-ish context to cut noise from the bare ticker.
        query = f'"{ticker.upper()}" (stock OR shares OR earnings OR nasdaq OR nyse)'
        params = {
            "query": query,
            "mode": "ArticleList",
            "format": "json",
            "maxrecords": str(min(int(limit), 250)),
            "sort": "DateDesc",
        }
        data = http.get_json(ENDPOINT, params=params,
                             headers={"User-Agent": "trading-bot-info/1.0"})
        articles = data.get("articles", []) if isinstance(data, dict) else []
        out: list[Item] = []
        for a in articles:
            url = a.get("url", "")
            title = a.get("title", "")
            if not (url or title):
                continue
            out.append(Item(
                id=make_id(self.name, url=url, headline=title),
                tickers=[ticker.upper()],
                published_utc=to_utc(a.get("seendate")),
                source=f"gdelt/{a.get('domain', '')}".rstrip("/"),
                item_type=NEWS,
                headline=title,
                summary="",
                url=url,
                extra={"domain": a.get("domain"), "language": a.get("language"),
                       "sourcecountry": a.get("sourcecountry")},
            ))
        return out
