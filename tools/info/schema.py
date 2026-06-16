"""
Normalized schema for the real-time information tool.

Every adapter, regardless of source, returns a list of `Item`s in this one shape so
the retriever can merge/dedupe/sort them uniformly. This mirrors the project's
"one interface, many adapters" convention (see data/loader.py).

    Item(id, tickers, published_utc, source, item_type, headline, summary, url, extra)

  * item_type in {"news", "filing", "analyst"}.
  * published_utc is a timezone-aware UTC datetime (or None if a source can't
    provide one, e.g. a current analyst price-target snapshot).
  * extra holds type-specific structured fields, e.g.
      filing  -> {"form","cik","accession"}
      analyst -> {"firm","action","from_grade","to_grade","price_target"}
    and {"scraped": True} flags items that came from HTML scraping rather than a
    structured feed/API.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

NEWS = "news"
FILING = "filing"
ANALYST = "analyst"
ITEM_TYPES = (NEWS, FILING, ANALYST)


@dataclass
class Item:
    id: str
    tickers: list[str]
    published_utc: Optional[datetime]
    source: str
    item_type: str
    headline: str
    summary: str = ""
    url: str = ""
    extra: dict = field(default_factory=dict)

    # -- serialization (used by the cache) --------------------------------- #
    def to_dict(self) -> dict:
        d = asdict(self)
        d["published_utc"] = self.published_utc.isoformat() if self.published_utc else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        ts = d.get("published_utc")
        return cls(
            id=d["id"],
            tickers=list(d.get("tickers") or []),
            published_utc=datetime.fromisoformat(ts) if ts else None,
            source=d.get("source", ""),
            item_type=d.get("item_type", ""),
            headline=d.get("headline", ""),
            summary=d.get("summary", "") or "",
            url=d.get("url", "") or "",
            extra=dict(d.get("extra") or {}),
        )


# --------------------------------------------------------------------------- #
# helpers shared by adapters
# --------------------------------------------------------------------------- #
def to_utc(dt) -> Optional[datetime]:
    """Coerce a datetime/epoch/ISO string to a timezone-aware UTC datetime."""
    if dt is None or dt == "":
        return None
    if isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt, tz=timezone.utc)
    if isinstance(dt, str):
        dt = _parse_dt_string(dt)
        if dt is None:
            return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _parse_dt_string(s: str) -> Optional[datetime]:
    s = s.strip()
    if not s:
        return None
    # ISO 8601 (handle trailing Z)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    # GDELT compact form: 20240131T153000Z
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # RFC 822 (RSS pubDate), e.g. "Mon, 31 Jan 2024 15:30:00 GMT"
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        return None


def make_id(source: str, native_id=None, url: str = "", headline: str = "") -> str:
    """Stable id: prefer a source-native id, else hash url, else hash headline."""
    if native_id not in (None, ""):
        return f"{source}:{native_id}"
    basis = (url or headline or "").strip().lower()
    digest = hashlib.sha1(f"{source}|{basis}".encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_headline(headline: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy dedup."""
    h = _PUNCT.sub(" ", (headline or "").lower())
    return _WS.sub(" ", h).strip()


def normalize_url(url: str) -> str:
    """Canonicalize a URL for exact dedup: drop scheme case, query, fragment, trailing /."""
    if not url:
        return ""
    u = url.strip()
    u = re.sub(r"#.*$", "", u)        # fragment
    u = re.sub(r"\?.*$", "", u)       # query string
    u = re.sub(r"/+$", "", u)         # trailing slashes
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE).lower()
    return u
