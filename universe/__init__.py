"""
Dynamic universe DISCOVERY (Phase 3.5).

Once per day a DETERMINISTIC screener over real Alpaca market data proposes new
high-risk/high-reward candidates; the existing LLM scorer EVALUATES them (it never
invents tickers); a deterministic gate re-confirms tradability + a hard liquidity
floor and has final say; then the persisted universe expands and the existing AI
rebalance allocates within it.

    from universe.store import UniverseStore
    from universe.screener import screen, ScreenConfig
    from universe.discovery import run_discovery

See UNIVERSE.md for the full flow, screen criteria, gates, the speculative-sleeve
model, and honest caveats.
"""

from .store import UniverseStore, CORE, SPECULATIVE

__all__ = ["UniverseStore", "CORE", "SPECULATIVE"]
