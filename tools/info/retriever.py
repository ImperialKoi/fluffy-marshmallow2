"""
Unified retriever: the single entry point an agent calls.

    get_info(ticker, types=None, limit=50, sources=None, as_of=None, use_cache=True)
        -> list[Item]   (newest first, deduplicated)

Responsibilities:
  * Merge enabled sources, filtered to the requested item `types`.
  * Per-source caching (short TTL) so rapid repeat calls don't hammer feeds.
  * RESILIENCE: each source runs in isolation — if one raises (or is unavailable
    for lack of keys), it is logged and skipped; the call still returns the rest.
  * `as_of`: drop anything published after that UTC instant (cheap, future-proofs
    backtesting). Items with an unknown timestamp are kept (can't prove they're future).
  * Dedup by canonical URL and by fuzzy headline match.
  * Sort newest first; undated items sort last.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .schema import Item, ITEM_TYPES, normalize_url, normalize_headline
from .store import default_cache
from . import sources as sources_mod

log = logging.getLogger("info_tool.retriever")

_FUZZY_THRESHOLD = 0.90


def get_info(ticker, types=None, limit=50, sources=None, as_of=None,
             use_cache=True, cache=None) -> list[Item]:
    if not ticker or not str(ticker).strip():
        raise ValueError("ticker is required")
    ticker = str(ticker).strip().upper()

    type_set = _validate_types(types)
    as_of = _coerce_as_of(as_of)
    cache = cache or default_cache()
    src_objs = sources_mod.build(sources)

    merged: list[Item] = []
    for src in src_objs:
        # skip sources that can't serve any requested type
        if type_set and not (set(src.item_types) & type_set):
            continue
        if not src.available():
            log.info("source '%s' unavailable: %s", src.name, src.unavailable_reason())
            continue
        try:
            got = _fetch_source(src, ticker, limit, use_cache, cache)
            merged.extend(got)
        except Exception as e:  # noqa: BLE001 — one bad source must not break the call
            log.warning("source '%s' failed for %s: %s", src.name, ticker, e)
            continue

    # filter by type
    if type_set:
        merged = [it for it in merged if it.item_type in type_set]

    # as_of filter (keep undated items — can't prove they're in the future)
    if as_of is not None:
        merged = [it for it in merged
                  if it.published_utc is None or it.published_utc <= as_of]

    merged = _dedupe(merged)
    merged.sort(key=_sort_key, reverse=True)
    return merged[: int(limit)] if limit else merged


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _fetch_source(src, ticker, limit, use_cache, cache) -> list[Item]:
    if use_cache:
        cached = cache.get(src.name, ticker)
        if cached is not None:
            log.debug("cache hit: %s/%s (%d items)", src.name, ticker, len(cached))
            return cached
    got = src.fetch(ticker, limit=limit) or []
    cache.put(src.name, ticker, got)
    return got


def _validate_types(types):
    if not types:
        return None
    if isinstance(types, str):
        types = [types]
    type_set = set(types)
    bad = type_set - set(ITEM_TYPES)
    if bad:
        raise ValueError(f"Unknown item type(s) {bad}. Valid: {ITEM_TYPES}")
    return type_set


def _coerce_as_of(as_of):
    if as_of is None:
        return None
    if isinstance(as_of, str):
        from .schema import to_utc
        return to_utc(as_of)
    if isinstance(as_of, datetime):
        return as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    raise TypeError("as_of must be a datetime, ISO string, or None")


def _sort_key(it: Item):
    # undated items sort last (treated as the oldest possible)
    return it.published_utc or datetime.min.replace(tzinfo=timezone.utc)


def _dedupe(items: list[Item]) -> list[Item]:
    """Remove duplicates by canonical URL, then by fuzzy headline within item_type.

    Process newest-first so the surviving copy is the most recent. Keeping fuzzy
    matching scoped to the same item_type avoids collapsing, say, a news headline
    into a near-identical filing title.
    """
    from difflib import SequenceMatcher

    ordered = sorted(items, key=_sort_key, reverse=True)
    seen_urls: set[str] = set()
    kept: list[Item] = []
    kept_norm: list[tuple[str, str]] = []  # (item_type, normalized_headline)

    for it in ordered:
        url_key = normalize_url(it.url)
        if url_key and url_key in seen_urls:
            continue
        norm = normalize_headline(it.headline)
        is_dup = False
        if norm:
            for itype, prev in kept_norm:
                if itype != it.item_type:
                    continue
                if norm == prev or SequenceMatcher(None, norm, prev).ratio() >= _FUZZY_THRESHOLD:
                    is_dup = True
                    break
        if is_dup:
            continue
        if url_key:
            seen_urls.add(url_key)
        if norm:
            kept_norm.append((it.item_type, norm))
        kept.append(it)
    return kept
