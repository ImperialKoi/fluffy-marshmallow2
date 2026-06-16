"""
Hype tracker tests — mocked news + price data, no live network.

Covers the scoring math of each always-on component, ranking order, and the key
robustness property: a missing/failing source must NOT crash the score; only the
available components are weighted and the result records which were used vs missing.

Run:  python tests/test_hype.py
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from signals.hype import HypeTracker

NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _news(n_recent, n_old, now=NOW):
    """n_recent items in the last 24h + n_old items spread over the prior 6 days."""
    items = [SimpleNamespace(published_utc=now - timedelta(hours=1)) for _ in range(n_recent)]
    for k in range(n_old):
        items.append(SimpleNamespace(published_utc=now - timedelta(days=2 + (k % 5))))
    return items


def _bars(vol_today_mult=1.0, jump=0.0, n=40, noise=0.005):
    """n bars with a small, nonzero baseline volatility (so the z-score baseline is
    well-defined); last bar volume = mult x baseline, last return = `jump`."""
    import numpy as np
    rng = np.random.default_rng(0)
    closes = [100.0]
    for r in rng.normal(0.0, noise, n - 1):
        closes.append(closes[-1] * (1.0 + r))
    if jump:
        closes[-1] = closes[-2] * (1.0 + jump)        # a clean spike on the last bar
    vols = [100.0] * (n - 1) + [100.0 * vol_today_mult]
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": vols}, index=idx)


def _tracker(news_fn=None, bars_fn=None):
    return HypeTracker(news_fn=news_fn or (lambda s: _news(10, 5)),
                       bars_fn=bars_fn or (lambda s: _bars(3.0, 0.10)),
                       rvol_window=20, price_window=20, news_baseline_days=7)


class TestComponentMath(unittest.TestCase):
    def test_relative_volume(self):
        t = _tracker(bars_fn=lambda s: _bars(vol_today_mult=3.0))
        r = t.score("AAPL", now=NOW)
        # rvol = 3 -> saturate(3) = 3/4 = 0.75
        self.assertAlmostEqual(r["components"]["rel_volume"], 0.75, places=3)
        self.assertAlmostEqual(r["raw"]["rel_volume"]["rvol"], 3.0, places=3)

    def test_price_move_large_z(self):
        t = _tracker(bars_fn=lambda s: _bars(jump=0.10))   # a 10% jump on flat history
        r = t.score("AAPL", now=NOW)
        self.assertGreater(r["components"]["price_move"], 0.9)
        self.assertGreater(r["raw"]["price_move"]["z_score"], 5)

    def test_news_velocity_ratio(self):
        # 10 in last 24h, 5 older over 7d baseline -> in_window=15, rate=15/7=2.14,
        # ratio = 10 / 2.14 = 4.67 -> saturate = 4.67/5.67 = 0.823
        t = _tracker(news_fn=lambda s: _news(10, 5))
        r = t.score("AAPL", now=NOW)
        self.assertAlmostEqual(r["raw"]["news_velocity"]["velocity_ratio"], 4.667, places=2)
        self.assertAlmostEqual(r["components"]["news_velocity"], 0.823, places=2)

    def test_score_in_unit_interval_and_weighted(self):
        t = _tracker()
        r = t.score("AAPL", now=NOW)
        self.assertTrue(0.0 <= r["score"] <= 1.0)
        self.assertEqual(set(r["used"]), {"news_velocity", "rel_volume", "price_move"})
        # equal weights -> score is the mean of the three components
        comps = r["components"]
        self.assertAlmostEqual(r["score"], sum(comps.values()) / 3.0, places=6)


class TestRanking(unittest.TestCase):
    def test_rank_orders_by_score(self):
        def bars_fn(sym):
            return _bars(vol_today_mult=5.0 if sym == "HOT" else 1.0,
                         jump=0.15 if sym == "HOT" else 0.0)
        t = _tracker(news_fn=lambda s: _news(8 if s == "HOT" else 1, 5), bars_fn=bars_fn)
        ranked = t.rank(["COLD", "HOT", "MID"], now=NOW)
        self.assertEqual(ranked[0]["symbol"], "HOT")
        scores = [r["score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestRobustness(unittest.TestCase):
    def test_failing_news_source_does_not_crash(self):
        def boom(sym):
            raise RuntimeError("news API down")
        t = _tracker(news_fn=boom)               # bars still work
        r = t.score("AAPL", now=NOW)
        self.assertIn("news_velocity", r["missing"])
        self.assertEqual(set(r["used"]), {"rel_volume", "price_move"})
        self.assertTrue(0.0 <= r["score"] <= 1.0)

    def test_missing_bars_only_news(self):
        t = _tracker(bars_fn=lambda s: None)
        r = t.score("AAPL", now=NOW)
        self.assertEqual(r["used"], ["news_velocity"])
        self.assertIn("rel_volume", r["missing"])
        self.assertIn("price_move", r["missing"])

    def test_all_sources_unavailable_score_is_nan_not_crash(self):
        def boom(sym):
            raise RuntimeError("down")
        t = _tracker(news_fn=boom, bars_fn=lambda s: None)
        r = t.score("AAPL", now=NOW)
        self.assertNotEqual(r["score"], r["score"])   # NaN
        self.assertEqual(r["used"], [])
        self.assertEqual(set(r["missing"]),
                         {"news_velocity", "rel_volume", "price_move", "google_trends", "social"})

    def test_optional_sources_off_by_default(self):
        t = _tracker()
        r = t.score("AAPL", now=NOW)
        self.assertIn("google_trends", r["missing"])
        self.assertIn("social", r["missing"])


class TestSnapshot(unittest.TestCase):
    def test_snapshot_writes_history(self):
        f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False); f.close()
        t = HypeTracker(news_fn=lambda s: _news(5, 5), bars_fn=lambda s: _bars(2.0, 0.05),
                        rvol_window=20, price_window=20, news_baseline_days=7,
                        history_csv=f.name)
        rows = t.snapshot(["AAPL", "MSFT"], now=NOW)
        self.assertEqual(len(rows), 2)
        with open(f.name) as fh:
            body = fh.read()
        self.assertIn("AAPL", body)
        self.assertIn("rel_volume", body)   # header present


if __name__ == "__main__":
    unittest.main(verbosity=2)
