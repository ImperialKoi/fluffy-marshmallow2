"""
Score-stability gate + new defaults — pure, no network.

Run:  python tests/test_stability.py
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents import stability
from portfolio.risk import PortfolioLimits


class TestVerdicts(unittest.TestCase):
    def test_stable(self):
        r = stability.analyze({"A": [0.3, 0.4, 0.5, 0.4]}, flip_threshold=2, min_runs=3)
        self.assertEqual(r["A"]["verdict"], "stable")
        self.assertEqual(r["A"]["sign_flips"], 0)

    def test_unstable_when_sign_keeps_flipping(self):
        r = stability.analyze({"A": [0.3, -0.2, 0.4, -0.1]}, flip_threshold=2, min_runs=3)
        self.assertEqual(r["A"]["verdict"], "unstable")
        self.assertGreaterEqual(r["A"]["sign_flips"], 2)

    def test_flipped_latest_only(self):
        r = stability.analyze({"A": [0.3, 0.4, 0.5, -0.1]}, flip_threshold=2, min_runs=3)
        self.assertEqual(r["A"]["verdict"], "flipped")   # one flip, but it's the latest

    def test_insufficient_history(self):
        r = stability.analyze({"A": [0.5]}, flip_threshold=2, min_runs=3)
        self.assertEqual(r["A"]["verdict"], "insufficient")

    def test_zeros_ignored_for_sign(self):
        # neutral 0 scores don't count as sign flips
        r = stability.analyze({"A": [0.3, 0.0, 0.4, 0.0, 0.5]}, flip_threshold=2, min_runs=3)
        self.assertEqual(r["A"]["verdict"], "stable")

    def test_unstable_symbols_filter(self):
        res = stability.analyze({
            "GOOD": [0.3, 0.4, 0.5],
            "BAD": [0.3, -0.3, 0.3, -0.3],
            "FLIP": [0.2, 0.2, -0.2],
        }, flip_threshold=2, min_runs=3)
        self.assertEqual(stability.unstable_symbols(res), ["BAD", "FLIP"])


class TestLoadRecentScores(unittest.TestCase):
    def test_loads_last_n_runs_and_skips_blanks(self):
        f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w");
        f.write("ts,symbol,score\n")
        # 3 runs at distinct timestamps; one blank score (halted) must be skipped
        f.write("2026-06-16T16:00:00+00:00,AAPL,0.5\n")
        f.write("2026-06-16T17:00:00+00:00,AAPL,\n")          # blank -> skipped
        f.write("2026-06-16T18:00:00+00:00,AAPL,-0.2\n")
        f.write("2026-06-16T18:00:00+00:00,MSFT,0.1\n")
        f.close()
        by_sym, runs = stability.load_recent_scores(decisions_csv=f.name, runs=5)
        self.assertEqual(by_sym["AAPL"], [0.5, -0.2])         # blank dropped
        self.assertIn("MSFT", by_sym)
        # 2 distinct non-blank run timestamps for AAPL
        self.assertEqual(len(runs), 2)


class TestNewDefaults(unittest.TestCase):
    def test_turnover_cap_default_wired(self):
        self.assertEqual(config.AI_TURNOVER_CAP, 0.20)
        self.assertEqual(PortfolioLimits.from_config().turnover_cap, 0.20)

    def test_warmup_config_present(self):
        self.assertTrue(getattr(config, "SERVICE_WARMUP_BARS", 0) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
