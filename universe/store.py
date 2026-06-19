"""
Persisted dynamic-universe store.

Holds the set of symbols the bot is allowed to trade, in two tiers:

  * CORE        — the pinned config universe (config.AI_UNIVERSE). Always present,
                  never evicted, always tier "core".
  * SPECULATIVE — names the daily discovery added. Each carries the discovery
                  rationale, the LLM conviction it was admitted on, an added_date,
                  and a last_seen timestamp (refreshed whenever the screener still
                  surfaces it), so churn/eviction is mechanical, not by eye.

The store is plain JSON on disk so the universe survives restarts. It places no
orders and reads no market data — it is a membership ledger. The deterministic GATE
(universe/discovery.py) is the only writer of dynamic entries; this module just
persists them and enforces the universe-size cap via a churn/eviction policy.

Eviction policy (when full and a stronger candidate wants in): never touch pinned
core names or names with an OPEN position (passed in as `protected`); among the rest,
drop the WEAKEST/STALEST first — lowest admitted conviction, oldest last_seen as the
tie-break. This keeps the freshest, highest-conviction speculative names.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import config

log = logging.getLogger("universe.store")

CORE = "core"
SPECULATIVE = "speculative"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UniverseStore:
    def __init__(self, path: str = None, pinned: list[str] = None):
        self.path = path or config.UNIVERSE_STORE_FILE
        self.pinned = [s.upper() for s in (pinned if pinned is not None else config.AI_UNIVERSE)]
        self._entries: dict[str, dict] = self._load()   # dynamic symbols only

    # -- persistence ------------------------------------------------------- #
    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                data = json.load(f)
            ent = data.get("entries", data) if isinstance(data, dict) else {}
            return {s.upper(): v for s, v in ent.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"updated_utc": _now(), "pinned": self.pinned,
                       "entries": self._entries}, f, indent=2)

    # -- membership -------------------------------------------------------- #
    def symbols(self) -> list[str]:
        """Full tradeable universe: pinned core + dynamic, deduped, order-stable."""
        return list(dict.fromkeys(self.pinned + sorted(self._entries)))

    def dynamic_symbols(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, symbol: str) -> bool:
        s = symbol.upper()
        return s in self.pinned or s in self._entries

    def tier(self, symbol: str) -> str:
        s = symbol.upper()
        if s in self.pinned:
            return CORE
        e = self._entries.get(s)
        return e.get("tier", SPECULATIVE) if e else CORE

    def tier_map(self) -> dict[str, str]:
        """symbol -> tier for the whole universe (pinned core + dynamic)."""
        m = {s: CORE for s in self.pinned}
        for s, e in self._entries.items():
            m[s] = e.get("tier", SPECULATIVE)
        return m

    def entry(self, symbol: str) -> dict:
        return dict(self._entries.get(symbol.upper(), {}))

    def entries(self) -> dict[str, dict]:
        return {s: dict(e) for s, e in self._entries.items()}

    def is_full(self, max_size: int = None) -> bool:
        max_size = max_size or config.UNIVERSE_MAX_SIZE
        return len(self.symbols()) >= max_size

    def free_slots(self, max_size: int = None) -> int:
        max_size = max_size or config.UNIVERSE_MAX_SIZE
        return max(0, max_size - len(self.symbols()))

    # -- mutation ---------------------------------------------------------- #
    def add(self, symbol: str, tier: str = SPECULATIVE, rationale: str = "",
            conviction: float = 0.0, screen_meta: dict = None) -> bool:
        """Add or refresh a dynamic symbol. Pinned core names are not stored here
        (they're always present); refreshing an existing entry bumps last_seen and
        keeps the original added_date. Returns True if newly added."""
        s = symbol.upper()
        if s in self.pinned:
            return False
        new = s not in self._entries
        e = self._entries.get(s, {})
        e["tier"] = tier
        e["rationale"] = (rationale or e.get("rationale", ""))[:300]
        e["conviction"] = float(conviction)
        e["last_seen"] = _now()
        if new:
            e["added_date"] = _today()
        if screen_meta:
            e["screen"] = screen_meta
        self._entries[s] = e
        return new

    def touch(self, symbol: str, conviction: float = None):
        """Refresh last_seen (and optionally conviction) for a still-surfacing name."""
        s = symbol.upper()
        if s in self._entries:
            self._entries[s]["last_seen"] = _now()
            if conviction is not None:
                self._entries[s]["conviction"] = float(conviction)

    def remove(self, symbol: str) -> bool:
        return self._entries.pop(symbol.upper(), None) is not None

    # -- churn / eviction -------------------------------------------------- #
    def eviction_candidates(self, protected: set = None) -> list[str]:
        """Dynamic symbols eligible for eviction, WEAKEST/STALEST first.

        Never includes pinned core names or `protected` symbols (e.g. open
        positions). Ordered by ascending conviction, then oldest last_seen."""
        protected = {s.upper() for s in (protected or set())}
        elig = [s for s in self._entries if s not in protected]
        elig.sort(key=lambda s: (self._entries[s].get("conviction", 0.0),
                                  self._entries[s].get("last_seen", "")))
        return elig

    def make_room(self, n: int = 1, protected: set = None,
                  max_size: int = None) -> list[str]:
        """Evict up to the weakest `n` symbols needed to keep room for `n` additions
        within max_size. Returns the symbols actually evicted."""
        max_size = max_size or config.UNIVERSE_MAX_SIZE
        evicted = []
        cands = self.eviction_candidates(protected)
        while self.free_slots(max_size) < n and cands:
            victim = cands.pop(0)
            self.remove(victim)
            evicted.append(victim)
            log.info("universe full -> evicted %s (weakest/stalest)", victim)
        return evicted
