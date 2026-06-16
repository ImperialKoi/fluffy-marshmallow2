"""
Alpaca broker wrapper.

A thin, safe layer over alpaca-py's TradingClient. The rest of the system talks
to *this*, not to Alpaca directly, so the live loop stays broker-agnostic and the
safety rules live in one place.

Safety model
------------
  * paper=True is the default everywhere. Paper trading uses simulated money.
  * Going live (real money) is intentionally hard: you must (1) construct with
    paper=False, AND (2) set env ALPACA_ALLOW_LIVE=yes. Missing either raises.
  * Keys are read from environment variables, never hardcoded:
        ALPACA_KEY,  ALPACA_SECRET           (paper keys for paper trading)
        ALPACA_LIVE_KEY, ALPACA_LIVE_SECRET  (only if you ever go live)

Get free paper keys at app.alpaca.markets -> Paper Trading -> API Keys.
"""

import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError


def _f(x):
    """Float-or-None helper for optional Alpaca position fields."""
    return float(x) if x is not None else None


class AlpacaBroker:
    def __init__(self, paper: bool = True):
        self.paper = paper

        if not paper:
            # Triple gate before real money is even reachable.
            if os.environ.get("ALPACA_ALLOW_LIVE", "").lower() != "yes":
                raise PermissionError(
                    "Refusing to start in LIVE mode. Set ALPACA_ALLOW_LIVE=yes to "
                    "confirm you understand this trades real money."
                )
            key = os.environ.get("ALPACA_LIVE_KEY")
            secret = os.environ.get("ALPACA_LIVE_SECRET")
        else:
            key = os.environ.get("ALPACA_KEY")
            secret = os.environ.get("ALPACA_SECRET")

        if not key or not secret:
            which = "ALPACA_LIVE_KEY/ALPACA_LIVE_SECRET" if not paper else "ALPACA_KEY/ALPACA_SECRET"
            raise EnvironmentError(f"Missing API keys. Set {which} in your environment.")

        self.client = TradingClient(key, secret, paper=paper)

    # ---- account / state ---------------------------------------------------
    def account_summary(self) -> dict:
        a = self.client.get_account()
        return {
            "mode": "PAPER" if self.paper else "LIVE",
            "status": str(a.status),
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "blocked": bool(a.trading_blocked or a.account_blocked),
        }

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def position_shares(self, symbol: str) -> int:
        """Current signed share count for a symbol (0 if no position)."""
        try:
            pos = self.client.get_open_position(symbol)
            return int(float(pos.qty))   # qty is positive for long, negative short
        except APIError:
            return 0                      # 404 == no open position

    def list_positions(self) -> list:
        """All open positions as normalized dicts (read-only).

        Returns one dict per position with the fields the Inventory needs:
        symbol, qty (signed: + long / - short), avg_entry_price, current_price,
        market_value, cost_basis, unrealized_pl, unrealized_plpc, side. Alpaca is
        the source of truth for these; we never mutate them here.
        """
        out = []
        for p in self.client.get_all_positions():
            out.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": _f(getattr(p, "current_price", None)),
                "market_value": _f(getattr(p, "market_value", None)),
                "cost_basis": _f(getattr(p, "cost_basis", None)),
                "unrealized_pl": _f(getattr(p, "unrealized_pl", None)),
                "unrealized_plpc": _f(getattr(p, "unrealized_plpc", None)),
                "side": str(getattr(p, "side", "")),
            })
        return out

    def latest_price(self, symbol: str) -> float:
        """Latest trade price via the data client (lazy import to keep deps light)."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        key = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_LIVE_KEY")
        secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_LIVE_SECRET")
        dc = StockHistoricalDataClient(key, secret)
        trade = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trade[symbol].price)

    # ---- orders -------------------------------------------------------------
    def submit_market_order(self, symbol: str, qty: int, side: OrderSide):
        """Submit a whole-share market order. qty must be > 0; side decides direction."""
        if qty <= 0:
            return None
        order = MarketOrderRequest(
            symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order_data=order)

    def liquidate(self, symbol: str):
        """Close any open position in a symbol."""
        try:
            return self.client.close_position(symbol)
        except APIError:
            return None

    def cancel_open_orders(self):
        return self.client.cancel_orders()
