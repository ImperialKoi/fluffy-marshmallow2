"""
Portfolio-level risk: configurable limits + a persisted drawdown kill switch.

The limits feed the deterministic constructor (portfolio/constructor.py). The kill
switch tracks the REAL (paper) account equity high-water mark across runs — like
live_trader's state file — and, once tripped, forces the book flat and blocks new
entries until it's explicitly reset. Risk is always on and deterministic; the LLM
cannot override it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import config


@dataclass
class PortfolioLimits:
    max_weight: float = 0.20          # max fraction of equity per name
    max_gross: float = 0.95           # max total invested
    min_cash: float = 0.05            # minimum cash buffer
    min_positions: int = 3            # diversification floor
    max_positions: int = 10           # cap on number of holdings
    min_score: float = 0.05           # secondary raw-score floor for a candidate
    weighting: str = "confidence"     # "equal" | "confidence"
    neutral_band: float = 0.10        # |conviction| <= band -> HOLD/trim, not exit
    turnover_cap: float = None        # optional per-rebalance sum|Δweight| cap (None=off)

    @classmethod
    def from_config(cls) -> "PortfolioLimits":
        return cls(
            max_weight=config.AI_MAX_WEIGHT, max_gross=config.AI_MAX_GROSS,
            min_cash=config.AI_MIN_CASH, min_positions=config.AI_MIN_POSITIONS,
            max_positions=config.AI_MAX_POSITIONS, min_score=config.AI_MIN_SCORE,
            weighting=config.AI_WEIGHTING,
            neutral_band=getattr(config, "AI_NEUTRAL_BAND", 0.10),
            turnover_cap=getattr(config, "AI_TURNOVER_CAP", None),
        )


class KillSwitch:
    """Persisted drawdown kill switch keyed on real account equity."""

    def __init__(self, max_drawdown: float = None, state_file: str = None):
        self.max_drawdown = (config.AI_DRAWDOWN_KILL if max_drawdown is None
                             else max_drawdown)
        self.state_file = state_file or config.AI_STATE_FILE
        self.state = self._load()

    def _load(self) -> dict:
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"peak_equity": 0.0, "halted": False}

    def save(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def update(self, equity: float) -> bool:
        """Record equity, trip the switch if drawdown breaches the limit. Returns
        True if currently halted. Once halted it stays halted until reset()."""
        self.state["peak_equity"] = max(self.state.get("peak_equity", 0.0), equity)
        peak = self.state["peak_equity"]
        if (self.max_drawdown is not None and peak > 0
                and (peak - equity) / peak >= self.max_drawdown):
            self.state["halted"] = True
        self.save()
        return self.halted

    @property
    def halted(self) -> bool:
        return bool(self.state.get("halted", False))

    def reset(self):
        self.state = {"peak_equity": 0.0, "halted": False}
        self.save()
