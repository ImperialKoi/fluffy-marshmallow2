"""
Cross-sectional (portfolio) strategy interface.

Deliberately NOT the single-asset `Strategy` ABC: a cross-sectional strategy needs
the WHOLE universe at once (to rank names against each other), and it emits a view
of every symbol, not a single +1/0/-1. The single-asset engine and strategies are
untouched; this is a parallel seam for Phase 3+.

Contract:
    evaluate(universe, as_of) -> PortfolioSignal
      * uses only information available AS OF `as_of` (no future leakage),
      * returns a per-symbol view {score in [-1,1], confidence in [0,1], rationale},
      * optionally an exposure_multiplier in [0,1] for whole-market risk-on/off.

The strategy is ADVISORY. A deterministic constructor turns these scores into
target weights, and a deterministic risk layer has final say over what trades
(see portfolio/constructor.py and portfolio/risk.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SymbolSignal:
    ticker: str
    score: float = 0.0          # [-1, 1]: + bullish / - bearish / 0 neutral
    confidence: float = 0.0     # [0, 1]
    rationale: str = ""
    ok: bool = True             # False if defaulted due to an error/empty input
    error: str = ""             # why it was defaulted, if so

    def clamped(self) -> "SymbolSignal":
        self.score = float(max(-1.0, min(1.0, self.score)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        return self


@dataclass
class PortfolioSignal:
    signals: dict[str, SymbolSignal]
    exposure_multiplier: float = 1.0       # [0,1] whole-portfolio risk scaler
    as_of: Optional[datetime] = None
    meta: dict = field(default_factory=dict)

    def scores(self) -> dict[str, float]:
        return {t: s.score for t, s in self.signals.items()}


class PortfolioStrategy(ABC):
    name: str = "portfolio-base"

    @abstractmethod
    def evaluate(self, universe: list[str], as_of: datetime = None) -> PortfolioSignal:
        """Return a per-symbol view of the whole universe as of `as_of`."""
        raise NotImplementedError
