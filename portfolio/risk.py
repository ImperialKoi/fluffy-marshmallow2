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


@dataclass
class SpeculativeLimits:
    """Risk EXTENSION for the riskier discovered universe — it only ever tightens.

    A hard cap on the COMBINED equity weight across all speculative/penny names (so a
    blowup in thin names can't sink the book), plus a tighter per-name weight cap and
    tighter stop/take-profit than core names. Core (pinned) names keep the normal
    PortfolioLimits; these caps apply only to symbols tagged tier 'speculative'.
    """
    enabled: bool = True
    sleeve_pct: float = 0.15          # max combined weight across all speculative names
    max_weight: float = 0.05          # tighter per-name cap (vs PortfolioLimits.max_weight)
    stop_pct: float = 0.04            # tight stop-loss
    take_profit_pct: float = 0.08     # tight take-profit

    @classmethod
    def from_config(cls) -> "SpeculativeLimits":
        return cls(
            enabled=getattr(config, "SPEC_SLEEVE_ENABLED", True),
            sleeve_pct=getattr(config, "SPEC_SLEEVE_PCT", 0.15),
            max_weight=getattr(config, "SPEC_MAX_WEIGHT", 0.05),
            stop_pct=getattr(config, "SPEC_STOP_PCT", 0.04),
            take_profit_pct=getattr(config, "SPEC_TAKE_PROFIT_PCT", 0.08),
        )


def enforce_speculative_sleeve(weights: dict, tier_map: dict,
                               spec: SpeculativeLimits = None) -> tuple[dict, list[str]]:
    """Deterministic risk layer (final say): tighten the constructor's weights for
    speculative names. Two passes, both purely REDUCING exposure (never loosening):

      1. cap each speculative name at spec.max_weight (excess -> cash),
      2. scale all speculative names down so their COMBINED weight <= spec.sleeve_pct.

    `tier_map` maps symbol -> 'core'|'speculative' (unknown -> 'core', untouched).
    Returns (adjusted_weights, notes). Core names are never modified here.
    """
    spec = spec or SpeculativeLimits.from_config()
    w = dict(weights)
    if not spec.enabled:
        return w, []
    tier_map = {k.upper(): v for k, v in (tier_map or {}).items()}
    spec_syms = [s for s in w if tier_map.get(s.upper()) == "speculative"]
    if not spec_syms:
        return w, []
    notes = []
    # 1. tighter per-name cap
    for s in spec_syms:
        if w[s] > spec.max_weight + 1e-9:
            notes.append(f"{s} spec per-name cap {w[s]:.2%}->{spec.max_weight:.2%}")
            w[s] = spec.max_weight
    # 2. combined sleeve cap
    spec_total = sum(w[s] for s in spec_syms)
    if spec_total > spec.sleeve_pct + 1e-9 and spec_total > 0:
        scale = spec.sleeve_pct / spec_total
        for s in spec_syms:
            w[s] *= scale
        notes.append(f"speculative sleeve {spec_total:.2%}->{spec.sleeve_pct:.2%} "
                     f"(x{scale:.2f}) — combined penny exposure capped")
    return w, notes


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
