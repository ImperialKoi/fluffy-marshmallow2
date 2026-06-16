"""
Source adapter interface. One subclass per source, behind this single interface so
the retriever can treat them uniformly (the same pattern data/loader.py uses for
price data).

A source declares:
  * name           short stable id, used as the cache key and `sources=[...]` selector
  * item_types     which Item.item_type values it can emit
  * default_on     whether it's included when the caller doesn't name sources
  * available()    cheap check that prereqs exist (keys/UA) — skipped if False
  * fetch()        do the work; return a list[Item]. May raise; the retriever
                   isolates failures so one broken source never breaks the call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import Item


class Source(ABC):
    name: str = "base"
    item_types: tuple = ()
    default_on: bool = True

    def available(self) -> bool:
        """Override when the source needs optional creds/config to function."""
        return True

    def unavailable_reason(self) -> str:
        return ""

    @abstractmethod
    def fetch(self, ticker: str, limit: int = 50) -> list[Item]:
        raise NotImplementedError
