"""
Tiny SQLite cache for fetched items, keyed by (source, ticker) with a short TTL.

Why: rapid repeated calls (an agent polling several tickers) shouldn't hammer the
feeds. Each (source, ticker) response is cached as a JSON payload alongside the
exact UTC fetch time; reads within TTL are served from disk. `use_cache=False`
bypasses reads (forcing a refetch), and `clear()` / `purge_expired()` manage it.

The cache is intentionally dumb: it stores whatever a source returned (a list of
Items). Cross-source merge/dedup/as_of all happen later in the retriever, so the
cache never has to understand item semantics.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from .schema import Item

DEFAULT_TTL_SECONDS = int(os.environ.get("INFO_CACHE_TTL", "600"))  # 10 minutes
DEFAULT_PATH = os.environ.get("INFO_CACHE_PATH", "results/info_cache.db")

_lock = threading.Lock()


class Cache:
    def __init__(self, path: str = DEFAULT_PATH, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.path = path
        self.ttl = ttl_seconds
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with _lock, self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS info_cache (
                    source       TEXT NOT NULL,
                    ticker       TEXT NOT NULL,
                    fetched_utc  TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (source, ticker)
                )
                """
            )

    # -- reads/writes ------------------------------------------------------ #
    def get(self, source: str, ticker: str) -> Optional[list[Item]]:
        """Return cached items if present AND within TTL, else None."""
        ticker = ticker.upper()
        with _lock, self._conn() as conn:
            row = conn.execute(
                "SELECT fetched_utc, payload_json FROM info_cache WHERE source=? AND ticker=?",
                (source, ticker),
            ).fetchone()
        if row is None:
            return None
        fetched = datetime.fromisoformat(row["fetched_utc"])
        age = (datetime.now(timezone.utc) - fetched).total_seconds()
        if age > self.ttl:
            return None
        data = json.loads(row["payload_json"])
        return [Item.from_dict(d) for d in data]

    def put(self, source: str, ticker: str, items: list[Item]) -> None:
        ticker = ticker.upper()
        payload = json.dumps([it.to_dict() for it in items])
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO info_cache (source, ticker, fetched_utc, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (source, ticker, now, payload),
            )

    # -- maintenance ------------------------------------------------------- #
    def clear(self, source: Optional[str] = None, ticker: Optional[str] = None) -> int:
        """Delete cache rows. No args clears everything. Returns rows deleted."""
        q = "DELETE FROM info_cache"
        conds, args = [], []
        if source:
            conds.append("source=?"); args.append(source)
        if ticker:
            conds.append("ticker=?"); args.append(ticker.upper())
        if conds:
            q += " WHERE " + " AND ".join(conds)
        with _lock, self._conn() as conn:
            cur = conn.execute(q, args)
            return cur.rowcount

    def purge_expired(self) -> int:
        with _lock, self._conn() as conn:
            rows = conn.execute("SELECT source, ticker, fetched_utc FROM info_cache").fetchall()
            stale = [
                (r["source"], r["ticker"]) for r in rows
                if (datetime.now(timezone.utc) - datetime.fromisoformat(r["fetched_utc"])).total_seconds() > self.ttl
            ]
            for src, tk in stale:
                conn.execute("DELETE FROM info_cache WHERE source=? AND ticker=?", (src, tk))
            return len(stale)


# process-wide default cache (lazily created)
_default: Optional[Cache] = None


def default_cache() -> Cache:
    global _default
    if _default is None:
        _default = Cache()
    return _default
