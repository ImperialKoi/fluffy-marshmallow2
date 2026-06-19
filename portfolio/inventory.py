"""
Portfolio inventory / state.

One reliable interface that reports exactly what is held at all times, over two
backends: a live/paper broker (Alpaca) and the backtest engine's simulated state.

SOURCE-OF-TRUTH RULE (critical)
-------------------------------
In paper/live, **Alpaca is authoritative** for quantities and cost basis. The
inventory SYNCS from the broker and never overrides it. The local metadata store
holds only things Alpaca doesn't keep (entry date, strategy/rationale tag, target
weight, stop level, hype-at-entry) plus an optional "expected qty" used purely for
reconciliation/sanity checks — it is never written back to the broker.

In backtest, the engine's simulated positions are the source; `SimBroker` adapts
that state to the same interface so the rest of the system is backend-agnostic.

This module is a read/report + local-log layer only. It places no orders and is
side-effect-free except for its own metadata DB and snapshot history.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import config

log = logging.getLogger("portfolio.inventory")
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# local metadata store (SQLite) — fields Alpaca does not keep
# --------------------------------------------------------------------------- #
META_FIELDS = ("entry_date", "strategy_tag", "rationale", "target_weight",
               "stop_level", "hype_at_entry", "expected_qty", "managed",
               "risk_tier", "added_date")

# SQL column type per metadata field (used to create/migrate the table).
_COLUMN_DDL = {
    "entry_date": "TEXT", "strategy_tag": "TEXT", "rationale": "TEXT",
    "target_weight": "REAL", "stop_level": "REAL", "hype_at_entry": "REAL",
    "expected_qty": "REAL", "managed": "INTEGER",
    # dynamic-universe discovery: risk tier (core|speculative) + when it joined.
    "risk_tier": "TEXT", "added_date": "TEXT",
}

# Safety default: a position with no local record is NOT managed — the Phase 3
# strategy/AI may only trade positions explicitly opted in (managed=True). This
# walls off anything discovered at sync that the bot has no record of.
DEFAULT_MANAGED = False


class MetadataStore:
    def __init__(self, path: str = None):
        self.path = path or config.PORTFOLIO_DB
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init()

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        with _lock, self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS position_meta (symbol TEXT PRIMARY KEY, "
                      "updated_utc TEXT)")
            # additive migration: add any metadata columns missing from an older DB
            have = {r[1] for r in c.execute("PRAGMA table_info(position_meta)")}
            for col in META_FIELDS:
                if col not in have:
                    c.execute(f"ALTER TABLE position_meta ADD COLUMN {col} {_COLUMN_DDL[col]}")

    def set(self, symbol: str, **fields):
        symbol = symbol.upper()
        unknown = set(fields) - set(META_FIELDS)
        if unknown:
            raise ValueError(f"Unknown metadata fields: {unknown}. Allowed: {META_FIELDS}")
        existing = self.get(symbol)
        existing.update(fields)
        cols = ["symbol"] + list(META_FIELDS) + ["updated_utc"]
        vals = [symbol] + [existing.get(f) for f in META_FIELDS] + \
               [datetime.now(timezone.utc).isoformat()]
        ph = ",".join("?" for _ in cols)
        with _lock, self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO position_meta ({','.join(cols)}) VALUES ({ph})", vals)

    def get(self, symbol: str) -> dict:
        with _lock, self._conn() as c:
            row = c.execute("SELECT * FROM position_meta WHERE symbol=?", (symbol.upper(),)).fetchone()
        if not row:
            return {}
        d = {k: row[k] for k in row.keys() if k != "symbol"}
        if d.get("managed") is not None:        # SQLite stores bools as 0/1
            d["managed"] = bool(d["managed"])
        return d

    def all(self) -> dict:
        with _lock, self._conn() as c:
            rows = c.execute("SELECT * FROM position_meta").fetchall()
        return {r["symbol"]: {k: r[k] for k in r.keys() if k != "symbol"} for r in rows}

    def delete(self, symbol: str):
        with _lock, self._conn() as c:
            c.execute("DELETE FROM position_meta WHERE symbol=?", (symbol.upper(),))


# --------------------------------------------------------------------------- #
# backtest adapter: present engine simulated state through the broker interface
# --------------------------------------------------------------------------- #
@dataclass
class SimPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float


class SimBroker:
    """Minimal broker-shaped source backed by simulated state (for backtests/tests).

    Implements the same read interface Inventory needs: list_positions() and
    account_summary(). Build it from the engine's current shares/cash/price.
    """

    def __init__(self, equity: float = None, cash: float = 0.0,
                 positions: list[SimPosition] = None):
        self._positions = list(positions or [])
        self._cash = float(cash)
        self._equity = equity

    def list_positions(self) -> list[dict]:
        out = []
        for p in self._positions:
            mv = p.qty * p.current_price
            out.append({
                "symbol": p.symbol, "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": mv, "cost_basis": p.qty * p.avg_entry_price,
                "unrealized_pl": mv - p.qty * p.avg_entry_price,
                "unrealized_plpc": None, "side": "long" if p.qty >= 0 else "short",
            })
        return out

    def account_summary(self) -> dict:
        mv = sum(p.qty * p.current_price for p in self._positions)
        equity = self._equity if self._equity is not None else self._cash + mv
        return {"mode": "BACKTEST", "status": "SIM", "equity": float(equity),
                "cash": float(self._cash), "buying_power": float(self._cash),
                "blocked": False}


def from_backtest(symbol: str, qty: float, avg_cost: float, last_price: float,
                  cash: float, equity: float = None, metadata_store=None,
                  history_path: str = None) -> "Inventory":
    """Convenience: an Inventory populated from single-asset engine state."""
    sim = SimBroker(equity=equity, cash=cash,
                    positions=[SimPosition(symbol, qty, avg_cost, last_price)])
    inv = Inventory(broker=sim, metadata_store=metadata_store, history_path=history_path)
    return inv.sync()


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
ZERO_HOLDING_KEYS = ("qty", "avg_cost", "cost_basis", "last_price", "market_value",
                     "weight", "unrealized_pl", "unrealized_pl_pct")


class Inventory:
    def __init__(self, broker=None, metadata_store: MetadataStore = None,
                 history_path: str = None, db_path: str = None):
        self.broker = broker
        self.meta = metadata_store or MetadataStore(db_path or config.PORTFOLIO_DB)
        self.history_path = history_path or config.PORTFOLIO_HISTORY_CSV
        self._positions: list[dict] = []
        self._account: dict = {}
        self.synced_at: Optional[datetime] = None
        # complete-freedom mode: the strategy may trade/sell/protect ANY position,
        # so the per-symbol `managed` wall-off is bypassed (see config.AI_FREE_TRADE).
        self.free_trade = getattr(config, "AI_FREE_TRADE", False)

    # -- sync from the source of truth ------------------------------------- #
    def sync(self) -> "Inventory":
        if self.broker is None:
            raise RuntimeError("Inventory has no broker/source to sync from.")
        self._positions = list(self.broker.list_positions() or [])
        self._account = dict(self.broker.account_summary() or {})
        self.synced_at = datetime.now(timezone.utc)
        log.info("synced %d positions (equity=%.2f, mode=%s)",
                 len(self._positions), self._account.get("equity", float("nan")),
                 self._account.get("mode", "?"))
        return self

    # -- per-symbol holdings ----------------------------------------------- #
    def holdings(self) -> dict[str, dict]:
        equity = self._equity()
        meta_all = self.meta.all()
        out = {}
        for p in self._positions:
            sym = p["symbol"].upper()
            out[sym] = self._compute(p, equity, meta_all.get(sym, {}))
        return out

    def _compute(self, p: dict, equity: float, meta: dict) -> dict:
        qty = float(p["qty"])
        avg = float(p["avg_entry_price"])
        last = p.get("current_price")
        if last is None and p.get("market_value") is not None and qty:
            last = p["market_value"] / qty
        last = float(last) if last is not None else avg
        cost_basis = qty * avg                       # signed; long -> positive
        market_value = qty * last
        unrealized_pl = market_value - cost_basis    # correct for long & short
        unrealized_pl_pct = (unrealized_pl / abs(cost_basis)) if cost_basis else 0.0
        weight = (market_value / equity) if equity else 0.0
        return {
            "symbol": p["symbol"].upper(),
            "qty": qty, "avg_cost": avg, "cost_basis": cost_basis,
            "last_price": last, "market_value": market_value,
            "weight": weight, "unrealized_pl": unrealized_pl,
            "unrealized_pl_pct": unrealized_pl_pct,
            "side": "long" if qty >= 0 else "short",
            # safety default: unrecorded positions are NOT managed by the strategy/AI
            "managed": bool(meta.get("managed", DEFAULT_MANAGED)),
            "metadata": meta,
        }

    def get(self, symbol: str) -> dict:
        """That symbol's holding, or a zeroed holding (still merging any metadata)."""
        symbol = symbol.upper()
        h = self.holdings().get(symbol)
        if h is not None:
            return h
        meta = self.meta.get(symbol)
        z = {"symbol": symbol, "side": "flat", "metadata": meta,
             "managed": bool(meta.get("managed", DEFAULT_MANAGED))}
        z.update({k: 0.0 for k in ZERO_HOLDING_KEYS})
        return z

    def is_managed(self, symbol: str) -> bool:
        """May the Phase 3 strategy/AI trade this symbol? In free-trade mode, YES for
        everything. Otherwise it defaults to False for any position with no local
        record (the safety wall-off)."""
        if self.free_trade:
            return True
        return bool(self.meta.get(symbol).get("managed", DEFAULT_MANAGED))

    def managed_symbols(self) -> list[str]:
        """Symbols explicitly opted in to strategy/AI trading."""
        return sorted(s for s, m in self.meta.all().items() if m.get("managed"))

    # -- portfolio totals -------------------------------------------------- #
    def totals(self) -> dict:
        hs = list(self.holdings().values())
        equity = self._equity()
        cash = float(self._account.get("cash", 0.0))
        mvs = [h["market_value"] for h in hs]
        gross = sum(abs(m) for m in mvs)
        net = sum(mvs)
        weights = [abs(h["weight"]) for h in hs]
        return {
            "equity": equity, "cash": cash,
            "mode": self._account.get("mode", "?"),
            "position_count": len(hs),
            "long_market_value": sum(m for m in mvs if m > 0),
            "short_market_value": sum(m for m in mvs if m < 0),
            "gross_exposure": gross, "net_exposure": net,
            "gross_exposure_pct": (gross / equity) if equity else 0.0,
            "net_exposure_pct": (net / equity) if equity else 0.0,
            "largest_weight": max(weights) if weights else 0.0,
            "total_unrealized_pl": sum(h["unrealized_pl"] for h in hs),
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
        }

    # -- reconciliation (flag, never override) ----------------------------- #
    def reconcile(self, expected: dict[str, float] = None) -> list[dict]:
        """Compare locally EXPECTED quantities vs broker ACTUAL. Returns a list of
        divergences and logs them. Never modifies broker state.

        `expected` maps symbol -> qty. If omitted, uses each symbol's stored
        `expected_qty` metadata. Divergence types:
          * qty_mismatch       expected and actual both nonzero but differ (partial fill)
          * missing_position   expected nonzero, actual zero
          * untracked_position actual nonzero, no expectation (manual trade / drift)
        """
        actual = {p["symbol"].upper(): float(p["qty"]) for p in self._positions}
        if expected is None:
            expected = {s: m["expected_qty"] for s, m in self.meta.all().items()
                        if m.get("expected_qty") is not None}
        expected = {s.upper(): float(q) for s, q in expected.items()}

        divergences = []
        for sym in sorted(set(actual) | set(expected)):
            a = actual.get(sym, 0.0)
            e = expected.get(sym)
            if e is None:
                if abs(a) > 1e-9:
                    divergences.append({"symbol": sym, "type": "untracked_position",
                                        "expected": None, "actual": a})
            elif abs(a - e) > 1e-6:
                typ = "missing_position" if abs(a) < 1e-9 else "qty_mismatch"
                divergences.append({"symbol": sym, "type": typ,
                                    "expected": e, "actual": a, "delta": a - e})
        for d in divergences:
            log.warning("RECONCILE divergence: %s", d)
        return divergences

    # -- snapshot history -------------------------------------------------- #
    def snapshot(self, note: str = "") -> dict:
        """Append a timestamped record of holdings + totals to the history log
        (CSV + SQLite). Returns the record. For the forward-test record."""
        import csv

        ts = datetime.now(timezone.utc).isoformat()
        hs = self.holdings()
        totals = self.totals()
        record = {"ts": ts, "note": note, "totals": totals, "holdings": hs}

        # 1) flat CSV: one row per holding + a __TOTALS__ row
        os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
        new = not os.path.exists(self.history_path)
        cols = ["ts", "symbol", "qty", "avg_cost", "last_price", "market_value",
                "weight", "unrealized_pl", "unrealized_pl_pct", "managed", "note"]
        with open(self.history_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            for sym, h in hs.items():
                w.writerow({k: h.get(k, "") for k in cols} | {"ts": ts, "symbol": sym, "note": note})
            w.writerow({"ts": ts, "symbol": "__TOTALS__",
                        "market_value": totals["gross_exposure"],
                        "weight": totals["net_exposure_pct"],
                        "unrealized_pl": totals["total_unrealized_pl"],
                        "qty": totals["position_count"],
                        "avg_cost": totals["cash"], "last_price": totals["equity"],
                        "unrealized_pl_pct": "", "note": note or "totals"})

        # 2) SQLite snapshot (full JSON blob, queryable by time)
        with _lock, sqlite3.connect(self.meta.path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS snapshots (
                ts TEXT, mode TEXT, equity REAL, cash REAL, gross_exposure REAL,
                net_exposure REAL, position_count INTEGER, note TEXT, payload_json TEXT)""")
            c.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                      (ts, totals["mode"], totals["equity"], totals["cash"],
                       totals["gross_exposure"], totals["net_exposure"],
                       totals["position_count"], note, json.dumps(record, default=str)))
        log.info("snapshot written (%d holdings) -> %s", len(hs), self.history_path)
        return record

    # -- helpers ----------------------------------------------------------- #
    def _equity(self) -> float:
        eq = self._account.get("equity")
        if eq:
            return float(eq)
        # fall back to cash + market value if the account didn't report equity
        cash = float(self._account.get("cash", 0.0))
        return cash + sum(float(p["qty"]) * float(p.get("current_price") or p["avg_entry_price"])
                          for p in self._positions)
