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

    # ---- tradable-asset universe (used by the dynamic-universe screener) -----
    def list_assets(self) -> list:
        """All ACTIVE, tradable US equities as normalized dicts (read-only).

        This is the screener's source of truth for tradability: a candidate that is
        not in this list is not tradable on Alpaca (many penny/OTC names are not) and
        is auto-excluded. Fields: symbol, name, exchange, tradable, status,
        fractionable, shortable, marginable, easy_to_borrow.
        """
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        out = []
        for a in self.client.get_all_assets(req):
            out.append(self._asset_dict(a))
        return out

    def get_asset(self, symbol: str) -> dict | None:
        """A single asset normalized dict, or None if Alpaca has no such asset.

        Used by the gate to RE-CONFIRM tradability right before a symbol joins the
        universe (an asset can be delisted/halted between the daily screen and now)."""
        try:
            return self._asset_dict(self.client.get_asset(symbol))
        except APIError:
            return None

    @staticmethod
    def _asset_dict(a) -> dict:
        return {
            "symbol": a.symbol,
            "name": getattr(a, "name", "") or "",
            "exchange": str(getattr(a, "exchange", "") or ""),
            "tradable": bool(getattr(a, "tradable", False)),
            "status": str(getattr(a, "status", "")),
            "fractionable": bool(getattr(a, "fractionable", False)),
            "shortable": bool(getattr(a, "shortable", False)),
            "marginable": bool(getattr(a, "marginable", False)),
            "easy_to_borrow": bool(getattr(a, "easy_to_borrow", False)),
        }

    def daily_bars(self, symbols, lookback_days: int = 40, feed: str = "iex") -> dict:
        """Recent daily OHLCV for MANY symbols at once -> {symbol: DataFrame}.

        One multi-symbol Alpaca request (the screener chunks large lists). Each frame
        is indexed by tz-naive date with open/high/low/close/volume columns, oldest
        first. Symbols with no data are simply absent from the result.
        """
        import pandas as pd
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        from datetime import datetime, timedelta, timezone

        syms = [s.upper() for s in symbols]
        if not syms:
            return {}
        key = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_LIVE_KEY")
        secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_LIVE_SECRET")
        dc = StockHistoricalDataClient(key, secret)
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        req = StockBarsRequest(
            symbol_or_symbols=syms, timeframe=TimeFrame.Day, start=start,
            feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP)
        df = dc.get_stock_bars(req).df
        out = {}
        if df is None or df.empty:
            return out
        df = df.reset_index()
        for sym, g in df.groupby("symbol"):
            g = g.set_index("timestamp")
            g.index = pd.to_datetime(g.index).tz_localize(None)
            out[str(sym)] = g[["open", "high", "low", "close", "volume"]].sort_index()
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

    # ---- protective / resting orders + clock (used by the always-on service) ---
    def get_open_orders(self) -> list:
        """Open orders as normalized dicts (read-only)."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        out = []
        for o in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            out.append({
                "id": str(o.id),
                "symbol": o.symbol,
                "qty": float(o.qty) if o.qty is not None else 0.0,
                "side": str(getattr(o, "side", "")),
                "type": str(getattr(o, "order_type", getattr(o, "type", ""))),
                "order_class": str(getattr(o, "order_class", "")),
                "stop_price": _f(getattr(o, "stop_price", None)),
                "limit_price": _f(getattr(o, "limit_price", None)),
                "trail_percent": _f(getattr(o, "trail_percent", None)),
            })
        return out

    def submit_stop_order(self, symbol: str, qty: int, stop_price: float,
                          side: OrderSide = OrderSide.SELL):
        """Server-side stop (market) order, GTC so it rests across sessions."""
        from alpaca.trading.requests import StopOrderRequest
        if qty <= 0:
            return None
        req = StopOrderRequest(symbol=symbol, qty=qty, side=side,
                               time_in_force=TimeInForce.GTC, stop_price=round(stop_price, 2))
        return self.client.submit_order(req)

    def submit_trailing_stop(self, symbol: str, qty: int, trail_percent: float,
                             side: OrderSide = OrderSide.SELL):
        """Server-side trailing stop, GTC. trail_percent is in PERCENT units (5.0 = 5%)."""
        from alpaca.trading.requests import TrailingStopOrderRequest
        if qty <= 0:
            return None
        req = TrailingStopOrderRequest(symbol=symbol, qty=qty, side=side,
                                       time_in_force=TimeInForce.GTC,
                                       trail_percent=round(trail_percent, 3))
        return self.client.submit_order(req)

    def submit_limit_order(self, symbol: str, qty: int, limit_price: float,
                           side: OrderSide = OrderSide.SELL):
        from alpaca.trading.requests import LimitOrderRequest
        if qty <= 0:
            return None
        req = LimitOrderRequest(symbol=symbol, qty=qty, side=side,
                                time_in_force=TimeInForce.GTC, limit_price=round(limit_price, 2))
        return self.client.submit_order(req)

    def submit_oco_exit(self, symbol: str, qty: int, take_profit_price: float,
                        stop_price: float):
        """OCO exit on an existing long: a take-profit limit + a stop, one cancels the
        other. Alpaca requires the take-profit as `take_profit.limit_price` and the stop
        as `stop_loss.stop_price` (not a top-level limit_price)."""
        from alpaca.trading.requests import (LimitOrderRequest, TakeProfitRequest,
                                             StopLossRequest)
        from alpaca.trading.enums import OrderClass
        if qty <= 0:
            return None
        tp = round(take_profit_price, 2)
        req = LimitOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
            order_class=OrderClass.OCO, limit_price=tp,
            take_profit=TakeProfitRequest(limit_price=tp),
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)))
        return self.client.submit_order(req)

    def cancel_order(self, order_id: str):
        try:
            return self.client.cancel_order_by_id(order_id)
        except APIError:
            return None

    def get_clock(self) -> dict:
        """Market clock: {is_open, next_open, next_close, timestamp} (regular hours)."""
        c = self.client.get_clock()
        return {"is_open": bool(c.is_open), "next_open": c.next_open,
                "next_close": c.next_close, "timestamp": c.timestamp}
