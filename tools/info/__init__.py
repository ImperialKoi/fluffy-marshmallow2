"""
Real-time market information tool.

A resilient, pluggable feed of recent news, SEC filings, and analyst ratings for a
ticker — the information layer a future AI decision agent reasons over. One
interface, many adapters (the same convention as data/loader.py), every source
failing soft.

    from tools.info import get_info, Item

    items = get_info("AAPL", types=["news", "filing", "analyst"], limit=30)
    for it in items:
        print(it.published_utc, it.item_type, it.source, it.headline)

See INFO_TOOL.md for sources, schema, rate limits, and the yfinance caveat.
"""

from .schema import Item, NEWS, FILING, ANALYST, ITEM_TYPES
from .retriever import get_info
from .store import Cache, default_cache
from . import sources

__all__ = ["get_info", "Item", "NEWS", "FILING", "ANALYST", "ITEM_TYPES",
           "Cache", "default_cache", "sources"]
