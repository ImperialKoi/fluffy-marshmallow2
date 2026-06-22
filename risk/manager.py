"""
Risk module.

This sits between the decision layer and execution and has veto power. Its job is
to turn an abstract signal (+1/0/-1) into a concrete share count, and to enforce
limits that protect the account regardless of what the strategy wants:

  * position sizing      - deploy only POSITION_FRACTION of equity per position
  * stop loss            - exit a losing trade at a fixed % below entry
  * take profit          - optionally lock in a fixed % gain
  * max-drawdown kill     - if the whole account draws down too far, stop trading

Keeping risk separate from the strategy is deliberate: it means a buggy or
over-optimistic strategy still can't blow up the account, and the same limits
apply identically in backtest, paper, and live.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskManager:
    position_fraction: float = 0.95
    stop_loss_pct: Optional[float] = 0.08
    take_profit_pct: Optional[float] = None
    max_drawdown_kill: Optional[float] = 0.25
    # MAXIMUM-RISK MODE: multiply every bullish/long position by this factor
    # (5x = quintuple the risk). Uses margin/leverage to buy 5x what it otherwise
    # would whenever the signal is bullish. 1.0 = normal sizing.
    risk_multiplier: float = 1.0

    # internal state
    _peak_equity: float = 0.0
    _halted: bool = False

    def target_shares(self, signal: int, equity: float, price: float) -> int:
        """How many shares to hold for the given signal (long-only sizing here;
        negative for shorts). Whole shares only — change to allow fractional.

        Bullish (long) signals are scaled by ``risk_multiplier`` so a bullish read
        buys that multiple of the normal budget (5x by default in max-risk mode)."""
        if self._halted or signal == 0 or price <= 0:
            return 0
        budget = equity * self.position_fraction
        if signal > 0:
            budget *= self.risk_multiplier   # quintuple the position on bullish patterns
        shares = int(budget // price)
        return shares if signal > 0 else -shares

    def stop_levels(self, entry_price: float, side: int):
        """Return (stop_price, take_profit_price) for a freshly opened position."""
        stop = tp = None
        if side > 0:  # long
            if self.stop_loss_pct is not None:
                stop = entry_price * (1 - self.stop_loss_pct)
            if self.take_profit_pct is not None:
                tp = entry_price * (1 + self.take_profit_pct)
        elif side < 0:  # short
            if self.stop_loss_pct is not None:
                stop = entry_price * (1 + self.stop_loss_pct)
            if self.take_profit_pct is not None:
                tp = entry_price * (1 - self.take_profit_pct)
        return stop, tp

    def check_stops(self, side: int, bar_high: float, bar_low: float,
                    stop: Optional[float], tp: Optional[float]):
        """
        Did an intraday stop / target trigger? Returns (triggered, fill_price).
        Conservative tie-breaking: if both could trigger in one bar, assume the
        stop hits first (worst case for the trader).
        """
        if side > 0:
            if stop is not None and bar_low <= stop:
                return True, stop
            if tp is not None and bar_high >= tp:
                return True, tp
        elif side < 0:
            if stop is not None and bar_high >= stop:
                return True, stop
            if tp is not None and bar_low <= tp:
                return True, tp
        return False, None

    def update_drawdown(self, equity: float) -> bool:
        """Track peak equity; halt trading if drawdown breaches the kill switch.
        Returns True if currently halted."""
        self._peak_equity = max(self._peak_equity, equity)
        if self.max_drawdown_kill is not None and self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd >= self.max_drawdown_kill:
                self._halted = True
        return self._halted

    @property
    def halted(self) -> bool:
        return self._halted
