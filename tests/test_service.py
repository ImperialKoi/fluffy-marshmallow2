"""
Always-on service tests — mocked broker + synthetic bars, NO live network.

Covers:
  * synthetic bars fill the buffer and drive the fast scan;
  * a managed position with no protective order -> protective stop placed;
  * reconcile is idempotent (no double-submission) and replaces a stale (qty-mismatch) order;
  * kill-switch trip -> go flat (cancel protective + market close);
  * market-hours gating: gated tasks don't run when the clock is closed;
  * both loops fire at their cadences (fake clock + short intervals);
  * an exception in one task does not crash the supervisor or stop the others.

Run:  python tests/test_service.py
"""

import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from service.buffer import BarBuffer
from service.clock import in_extended_window
from service.protective import ProtectiveOrderManager, ProtectiveSettings
from service.fast_scan import run_fast_scan, go_flat
from service.supervisor import Supervisor
from portfolio.risk import KillSwitch
from strategies.registry import build as build_strategy


# --------------------------------------------------------------------------- #
# mocks
# --------------------------------------------------------------------------- #
class MockBroker:
    def __init__(self, equity=100_000.0, positions=None, open_orders=None):
        self._equity = equity
        self._positions = positions or []
        self._open_orders = open_orders or []
        self.submitted = []
        self.canceled = []

    def equity(self):
        return self._equity

    def list_positions(self):
        return [dict(p) for p in self._positions]

    def get_open_orders(self):
        return [dict(o) for o in self._open_orders]

    def _record_order(self, kind, symbol, qty, **extra):
        oid = f"{kind}-{symbol}-{len(self._open_orders)}"
        o = {"id": oid, "symbol": symbol, "qty": float(qty), "side": "OrderSide.SELL",
             "type": kind, "order_class": "oco" if kind == "oco" else "", **extra}
        self._open_orders.append(o)
        self.submitted.append((kind, symbol, int(qty)))
        return o

    def submit_stop_order(self, symbol, qty, stop_price, side=None):
        return self._record_order("stop", symbol, qty, stop_price=stop_price)

    def submit_trailing_stop(self, symbol, qty, trail_percent, side=None):
        return self._record_order("trailing_stop", symbol, qty, trail_percent=trail_percent)

    def submit_oco_exit(self, symbol, qty, take_profit_price, stop_price):
        return self._record_order("oco", symbol, qty, limit_price=take_profit_price,
                                  stop_price=stop_price)

    def submit_market_order(self, symbol, qty, side):
        self.submitted.append(("market", symbol, int(qty)))
        return type("O", (), {"id": f"mkt-{symbol}"})()

    def cancel_order(self, oid):
        self.canceled.append(oid)
        self._open_orders = [o for o in self._open_orders if o["id"] != oid]
        return True


class MockInventory:
    def __init__(self, managed=None, qtys=None):
        self._m = {k.upper(): v for k, v in (managed or {}).items()}
        self._q = {k.upper(): v for k, v in (qtys or {}).items()}

    def is_managed(self, s):
        return self._m.get(s.upper(), False)

    def get(self, s):
        return {"qty": self._q.get(s.upper(), 0)}


class FakeClock:
    def __init__(self, open_=True):
        self._open = open_

    def is_open(self):
        return self._open


def _ks(max_dd=0.99, halted=False):
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False); f.close()
    ks = KillSwitch(max_drawdown=max_dd, state_file=f.name)
    if halted:
        ks.state["halted"] = True
    return ks


def _fill_buffer(buf, symbol, n=40, start=100.0, step=0.5):
    base = datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc)
    for i in range(n):
        px = start + i * step
        buf.add_bar(symbol, base + pd.Timedelta(minutes=i), px, px + 0.2, px - 0.2, px, 1000)


# --------------------------------------------------------------------------- #
# buffer + clock
# --------------------------------------------------------------------------- #
class TestBuffer(unittest.TestCase):
    def test_rolling_and_frame(self):
        buf = BarBuffer(maxlen=10)
        _fill_buffer(buf, "AAPL", n=25)
        self.assertEqual(buf.n("AAPL"), 10)            # bounded
        df = buf.frame("AAPL")
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 10)
        self.assertEqual(buf.latest_price("AAPL"), df["close"].iloc[-1])
        self.assertIsNone(buf.frame("ZZZZ"))


class TestExtendedWindow(unittest.TestCase):
    def test_windows(self):
        # Tue 2026-06-16 08:00 ET = 12:00 UTC -> pre-market window
        self.assertTrue(in_extended_window(datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)))
        # Tue 13:00 ET = 17:00 UTC -> regular hours -> NOT an extended window
        self.assertFalse(in_extended_window(datetime(2026, 6, 16, 17, 0, tzinfo=timezone.utc)))
        # Sat -> never
        self.assertFalse(in_extended_window(datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)))


# --------------------------------------------------------------------------- #
# protective orders
# --------------------------------------------------------------------------- #
class TestProtective(unittest.TestCase):
    def setUp(self):
        self.pm = ProtectiveOrderManager(ProtectiveSettings(enabled=True, stop_pct=0.08))
        self.inv = MockInventory(managed={"AAPL": True}, qtys={"AAPL": 100})

    def test_places_stop_when_missing(self):
        broker = MockBroker(positions=[{"symbol": "AAPL", "qty": 100, "avg_entry_price": 200.0}])
        actions = self.pm.reconcile(broker, self.inv, mode="paper")
        self.assertEqual(broker.submitted, [("stop", "AAPL", 100)])
        self.assertEqual(actions[0]["action"], "placed")
        self.assertAlmostEqual(actions[0]["stop_price"], 184.0)   # 200 * 0.92

    def test_idempotent_no_double_submit(self):
        broker = MockBroker(positions=[{"symbol": "AAPL", "qty": 100, "avg_entry_price": 200.0}])
        self.pm.reconcile(broker, self.inv, mode="paper")        # places
        self.pm.reconcile(broker, self.inv, mode="paper")        # already covered
        self.assertEqual(len(broker.submitted), 1)               # NOT submitted twice

    def test_replaces_stale_qty_mismatch(self):
        broker = MockBroker(
            positions=[{"symbol": "AAPL", "qty": 200, "avg_entry_price": 200.0}],
            open_orders=[{"id": "old", "symbol": "AAPL", "qty": 100.0,
                          "side": "OrderSide.SELL", "type": "stop", "order_class": "",
                          "stop_price": 184.0}])
        actions = self.pm.reconcile(broker, self.inv, mode="paper")
        self.assertIn("old", broker.canceled)                    # stale cancelled
        self.assertEqual(broker.submitted, [("stop", "AAPL", 200)])  # re-placed at new qty

    def test_dry_mode_places_nothing(self):
        broker = MockBroker(positions=[{"symbol": "AAPL", "qty": 100, "avg_entry_price": 200.0}])
        actions = self.pm.reconcile(broker, self.inv, mode="dry")
        self.assertEqual(broker.submitted, [])
        self.assertEqual(actions[0]["action"], "would_place")

    def test_unmanaged_position_not_protected(self):
        broker = MockBroker(positions=[{"symbol": "TSLA", "qty": 50, "avg_entry_price": 100.0}])
        inv = MockInventory(managed={"TSLA": False}, qtys={"TSLA": 50})
        actions = self.pm.reconcile(broker, inv, mode="paper")
        self.assertEqual(broker.submitted, [])                   # walled off
        self.assertEqual(actions, [])


# --------------------------------------------------------------------------- #
# fast scan
# --------------------------------------------------------------------------- #
class TestFastScan(unittest.TestCase):
    def test_synthetic_bars_drive_scan_and_protect(self):
        buf = BarBuffer(maxlen=240)
        _fill_buffer(buf, "AAPL", n=40)
        broker = MockBroker(positions=[{"symbol": "AAPL", "qty": 100, "avg_entry_price": 200.0}])
        inv = MockInventory(managed={"AAPL": True}, qtys={"AAPL": 100})
        pm = ProtectiveOrderManager(ProtectiveSettings(enabled=True, stop_pct=0.08))
        res = run_fast_scan(buffer=buf, broker=broker, inventory=inv, killswitch=_ks(),
                            protective=pm, fast_strategy=build_strategy("supertrend"),
                            universe=["AAPL"], mode="paper", min_bars=30)
        self.assertFalse(res["halted"])
        self.assertIn("AAPL", res["signals"])                    # bars produced a signal
        self.assertEqual(broker.submitted, [("stop", "AAPL", 100)])  # protected

    def test_kill_switch_goes_flat(self):
        broker = MockBroker(equity=70_000.0,
                            positions=[{"symbol": "AAPL", "qty": 100, "avg_entry_price": 200.0}],
                            open_orders=[{"id": "s1", "symbol": "AAPL", "qty": 100.0,
                                          "side": "OrderSide.SELL", "type": "stop"}])
        inv = MockInventory(managed={"AAPL": True}, qtys={"AAPL": 100})
        pm = ProtectiveOrderManager()
        res = run_fast_scan(buffer=BarBuffer(), broker=broker, inventory=inv,
                            killswitch=_ks(halted=True), protective=pm,
                            fast_strategy=build_strategy("supertrend"),
                            universe=["AAPL"], mode="paper", min_bars=30)
        self.assertTrue(res["halted"])
        self.assertIn("s1", broker.canceled)                     # protective cancelled
        self.assertIn(("market", "AAPL", 100), broker.submitted)  # flattened


# --------------------------------------------------------------------------- #
# supervisor: cadence, gating, exception isolation
# --------------------------------------------------------------------------- #
class TestSupervisor(unittest.TestCase):
    def _counts(self):
        return {"fast": 0, "slow": 0, "sync": 0}

    def test_both_loops_fire_at_cadence(self):
        c = self._counts()
        sup = Supervisor(clock=FakeClock(True), scan_interval=0.05, rebalance_interval=0.15,
                         sync_interval=0.05,
                         fast_fn=lambda: c.__setitem__("fast", c["fast"] + 1),
                         slow_fn=lambda: c.__setitem__("slow", c["slow"] + 1),
                         sync_fn=lambda: c.__setitem__("sync", c["sync"] + 1))
        asyncio.run(sup.run_for(0.5))
        self.assertGreaterEqual(c["fast"], 3)        # fires often
        self.assertGreaterEqual(c["slow"], 2)        # fires less often
        self.assertGreater(c["fast"], c["slow"])     # faster cadence -> more calls

    def test_market_closed_gates_loops(self):
        c = self._counts()
        sup = Supervisor(clock=FakeClock(False), scan_interval=0.05, rebalance_interval=0.05,
                         sync_interval=0.05,
                         fast_fn=lambda: c.__setitem__("fast", c["fast"] + 1),
                         slow_fn=lambda: c.__setitem__("slow", c["slow"] + 1),
                         sync_fn=lambda: c.__setitem__("sync", c["sync"] + 1))
        asyncio.run(sup.run_for(0.3))
        self.assertEqual(c["fast"], 0)               # gated off when closed
        self.assertEqual(c["slow"], 0)
        self.assertGreaterEqual(c["sync"], 1)        # ungated -> still runs

    def test_task_exception_does_not_crash_service(self):
        c = self._counts()

        def boom():
            c["fast"] += 1
            raise RuntimeError("kaboom")

        sup = Supervisor(clock=FakeClock(True), scan_interval=0.05, rebalance_interval=0.05,
                         sync_interval=0.05, fast_fn=boom,
                         slow_fn=lambda: c.__setitem__("slow", c["slow"] + 1),
                         sync_fn=lambda: c.__setitem__("sync", c["sync"] + 1))
        asyncio.run(sup.run_for(0.4))                # must return normally
        self.assertGreaterEqual(c["fast"], 2)        # kept being retried despite raising
        self.assertGreaterEqual(c["slow"], 2)        # peer task unaffected


if __name__ == "__main__":
    unittest.main(verbosity=2)
