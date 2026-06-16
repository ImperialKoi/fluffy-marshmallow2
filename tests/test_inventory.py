"""
Inventory tests — mocked broker only, no live network.

Headline fixture (from the task): broker reports 326 shares of AAPL at avg cost
$291.84 -> holdings() must show qty 326, cost_basis ~= $95,139.84, and compute
market_value / weight / unrealized P&L from the live price.

Run:  python tests/test_inventory.py
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.inventory import Inventory, MetadataStore, SimBroker, SimPosition, from_backtest


class MockBroker:
    """Broker-shaped stub: returns fixed positions + account."""
    def __init__(self, positions, equity, cash, mode="PAPER"):
        self._positions = positions
        self._equity = equity
        self._cash = cash
        self._mode = mode

    def list_positions(self):
        return [dict(p) for p in self._positions]

    def account_summary(self):
        return {"mode": self._mode, "status": "ACTIVE", "equity": self._equity,
                "cash": self._cash, "buying_power": self._cash, "blocked": False}


def _tmp():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False); f.close()
    return f.name


class TestAAPLFixture(unittest.TestCase):
    def setUp(self):
        # 326 AAPL @ $291.84, live price $320.00, account equity $200k.
        self.broker = MockBroker(
            positions=[{"symbol": "AAPL", "qty": 326.0, "avg_entry_price": 291.84,
                        "current_price": 320.00, "market_value": None,
                        "cost_basis": None, "unrealized_pl": None,
                        "unrealized_plpc": None, "side": "long"}],
            equity=200_000.0, cash=95_680.0)
        self.inv = Inventory(broker=self.broker, metadata_store=MetadataStore(_tmp()),
                             history_path=_tmp() + ".csv").sync()

    def test_holdings_math(self):
        h = self.inv.holdings()["AAPL"]
        self.assertEqual(h["qty"], 326)
        self.assertAlmostEqual(h["avg_cost"], 291.84, places=2)
        self.assertAlmostEqual(h["cost_basis"], 95_139.84, places=2)      # 326 * 291.84
        self.assertAlmostEqual(h["market_value"], 104_320.00, places=2)   # 326 * 320
        self.assertAlmostEqual(h["unrealized_pl"], 9_180.16, places=2)    # 104320 - 95139.84
        self.assertAlmostEqual(h["unrealized_pl_pct"], 9_180.16 / 95_139.84, places=4)
        self.assertAlmostEqual(h["weight"], 104_320.00 / 200_000.0, places=4)

    def test_totals(self):
        t = self.inv.totals()
        self.assertEqual(t["position_count"], 1)
        self.assertAlmostEqual(t["equity"], 200_000.0, places=2)
        self.assertAlmostEqual(t["gross_exposure"], 104_320.00, places=2)
        self.assertAlmostEqual(t["largest_weight"], 104_320.00 / 200_000.0, places=4)

    def test_get_absent_symbol_is_zeroed(self):
        z = self.inv.get("NFLX")
        self.assertEqual(z["qty"], 0.0)
        self.assertEqual(z["market_value"], 0.0)
        self.assertEqual(z["side"], "flat")


class TestReconcile(unittest.TestCase):
    def test_divergence_detection_without_override(self):
        broker = MockBroker(
            positions=[
                {"symbol": "AAPL", "qty": 326.0, "avg_entry_price": 291.84,
                 "current_price": 320.0, "side": "long"},
                {"symbol": "TSLA", "qty": 50.0, "avg_entry_price": 200.0,
                 "current_price": 210.0, "side": "long"},   # untracked (manual)
            ], equity=200_000.0, cash=50_000.0)
        meta = MetadataStore(_tmp())
        meta.set("AAPL", expected_qty=326)     # matches -> no divergence
        meta.set("MSFT", expected_qty=100)     # expected but not held -> missing
        # TSLA held but no expectation -> untracked
        inv = Inventory(broker=broker, metadata_store=meta, history_path=_tmp() + ".csv").sync()

        div = inv.reconcile()
        types = {d["symbol"]: d["type"] for d in div}
        self.assertNotIn("AAPL", types)                       # reconciles cleanly
        self.assertEqual(types.get("MSFT"), "missing_position")
        self.assertEqual(types.get("TSLA"), "untracked_position")
        # broker state must be untouched (no override)
        self.assertEqual(inv.holdings()["AAPL"]["qty"], 326)

    def test_partial_fill_is_qty_mismatch(self):
        broker = MockBroker(
            positions=[{"symbol": "AAPL", "qty": 200.0, "avg_entry_price": 291.84,
                        "current_price": 320.0, "side": "long"}],
            equity=100_000.0, cash=40_000.0)
        meta = MetadataStore(_tmp()); meta.set("AAPL", expected_qty=326)
        inv = Inventory(broker=broker, metadata_store=meta, history_path=_tmp() + ".csv").sync()
        div = inv.reconcile()
        self.assertEqual(len(div), 1)
        self.assertEqual(div[0]["type"], "qty_mismatch")
        self.assertAlmostEqual(div[0]["delta"], 200 - 326)


class TestSnapshotAndMetadata(unittest.TestCase):
    def test_snapshot_writes_history(self):
        broker = MockBroker(
            positions=[{"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0,
                        "current_price": 110.0, "side": "long"}],
            equity=10_000.0, cash=8_900.0)
        hist = _tmp() + ".csv"
        inv = Inventory(broker=broker, metadata_store=MetadataStore(_tmp()),
                        history_path=hist).sync()
        rec = inv.snapshot(note="unit")
        self.assertTrue(os.path.exists(hist))
        with open(hist) as f:
            body = f.read()
        self.assertIn("AAPL", body)
        self.assertIn("__TOTALS__", body)
        self.assertEqual(rec["totals"]["position_count"], 1)

    def test_metadata_roundtrip_and_merge(self):
        meta = MetadataStore(_tmp())
        meta.set("AAPL", entry_date="2026-01-02", strategy_tag="supertrend",
                 hype_at_entry=0.71, target_weight=0.05)
        meta.set("AAPL", stop_level=280.0)   # merge, not overwrite
        got = meta.get("AAPL")
        self.assertEqual(got["strategy_tag"], "supertrend")
        self.assertEqual(got["stop_level"], 280.0)
        self.assertEqual(got["hype_at_entry"], 0.71)
        with self.assertRaises(ValueError):
            meta.set("AAPL", bogus_field=1)


class TestBacktestAdapter(unittest.TestCase):
    def test_sim_broker_same_interface(self):
        inv = from_backtest("AAPL", qty=100, avg_cost=150.0, last_price=165.0,
                            cash=5_000.0, metadata_store=MetadataStore(_tmp()),
                            history_path=_tmp() + ".csv")
        h = inv.holdings()["AAPL"]
        self.assertEqual(h["qty"], 100)
        self.assertAlmostEqual(h["market_value"], 16_500.0)
        self.assertAlmostEqual(h["unrealized_pl"], 1_500.0)        # (165-150)*100
        t = inv.totals()
        self.assertEqual(t["mode"], "BACKTEST")
        self.assertAlmostEqual(t["equity"], 5_000.0 + 16_500.0)    # cash + MV


if __name__ == "__main__":
    unittest.main(verbosity=2)
