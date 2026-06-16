"""
Phase 3 AI portfolio tests — mocked LLM + mocked broker, NO live network.

Covers: LLM JSON parsing/validation, graceful handling of malformed/empty/
out-of-universe LLM output, constructor weighting + limit enforcement, multi-symbol
reconciliation (incl. the unmanaged-position wall-off and new-entry flag), and the
benchmark-tracking math.

Run:  python tests/test_ai_portfolio.py
"""

import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm import LLM, LLMResult, extract_json, StubLLM
from agents.news_portfolio import NewsPortfolioStrategy
from strategies.portfolio_base import SymbolSignal
from portfolio.constructor import construct_targets
from portfolio.risk import PortfolioLimits
from live_portfolio import (compute_orders, equal_weight_benchmark, spy_benchmark,
                            portfolio_turnover)

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# mocks
# --------------------------------------------------------------------------- #
class MockLLM(LLM):
    """Returns a preset score per ticker (parsed from the prompt), or a forced
    response for malformed/edge-case tests."""
    name = "mock"

    def __init__(self, mapping=None, force=None):
        self.mapping = mapping or {}
        self.force = force

    def complete_json(self, prompt: str) -> LLMResult:
        if self.force is not None:
            return self.force(prompt)
        m = re.search(r"TICKER:\s*(\w+)", prompt)
        sym = m.group(1) if m else ""
        if sym in self.mapping:
            score, conf = self.mapping[sym]
            parsed = {"ticker": sym, "score": score, "confidence": conf,
                      "rationale": "mock rationale"}
            return LLMResult(parsed=parsed, raw=json.dumps(parsed))
        return LLMResult(parsed=None, raw="", error="no mapping")


def fake_get_info(n=3):
    def _fn(sym, as_of=None, limit=12):
        return [{"item_type": "news", "published_utc": datetime(2026, 6, 15, tzinfo=UTC),
                 "headline": f"{sym} headline {i}", "summary": "details",
                 "source": "rss/test"} for i in range(n)]
    return _fn


class MockBroker:
    def __init__(self, positions, equity):
        self._pos = positions          # symbol -> qty
        self._equity = equity
        self.orders = []

    def equity(self):
        return self._equity

    def list_positions(self):
        return [{"symbol": s, "qty": q} for s, q in self._pos.items() if q]


# --------------------------------------------------------------------------- #
# LLM parsing / validation
# --------------------------------------------------------------------------- #
class TestLLMParsing(unittest.TestCase):
    def test_extract_json_plain_and_fenced(self):
        self.assertEqual(extract_json('{"a": 1}')["a"], 1)
        self.assertEqual(extract_json('```json\n{"a": 2}\n```')["a"], 2)
        self.assertEqual(extract_json("prefix {\"a\": 3} suffix")["a"], 3)
        self.assertIsNone(extract_json("not json at all"))
        self.assertIsNone(extract_json(""))

    def test_stub_llm_is_deterministic_structured(self):
        stub = StubLLM()
        r1 = stub.complete_json("TICKER: AAPL\nAAPL beats record upgrade surge")
        r2 = stub.complete_json("TICKER: AAPL\nAAPL beats record upgrade surge")
        self.assertEqual(r1.parsed, r2.parsed)
        self.assertGreater(r1.parsed["score"], 0)         # bullish words -> positive
        bear = stub.complete_json("TICKER: X\nlawsuit downgrade plunge miss recall")
        self.assertLess(bear.parsed["score"], 0)


class TestStrategyValidation(unittest.TestCase):
    UNIVERSE = ["AAPL", "MSFT", "TSLA"]

    def _strategy(self, llm):
        return NewsPortfolioStrategy(llm=llm, universe=self.UNIVERSE,
                                     get_info_fn=fake_get_info(), audit_dir=tempfile.mkdtemp())

    def test_valid_scores_parsed(self):
        llm = MockLLM(mapping={"AAPL": (0.8, 0.9), "MSFT": (-0.5, 0.6), "TSLA": (0.1, 0.3)})
        ps = self._strategy(llm).evaluate(self.UNIVERSE, as_of=datetime(2026, 6, 15, tzinfo=UTC))
        self.assertAlmostEqual(ps.signals["AAPL"].score, 0.8)
        self.assertTrue(ps.signals["AAPL"].ok)
        self.assertAlmostEqual(ps.signals["MSFT"].score, -0.5)

    def test_out_of_range_is_clamped(self):
        llm = MockLLM(mapping={"AAPL": (5.0, 9.0), "MSFT": (-3.0, -1.0), "TSLA": (0.0, 0.0)})
        ps = self._strategy(llm).evaluate(self.UNIVERSE)
        self.assertEqual(ps.signals["AAPL"].score, 1.0)        # clamped to [-1,1]
        self.assertEqual(ps.signals["AAPL"].confidence, 1.0)   # clamped to [0,1]
        self.assertEqual(ps.signals["MSFT"].score, -1.0)
        self.assertEqual(ps.signals["MSFT"].confidence, 0.0)

    def test_malformed_defaults_to_zero(self):
        llm = MockLLM(force=lambda p: LLMResult(parsed=None, raw="garbage", error="bad"))
        ps = self._strategy(llm).evaluate(self.UNIVERSE)
        for sym in self.UNIVERSE:
            self.assertEqual(ps.signals[sym].score, 0.0)
            self.assertFalse(ps.signals[sym].ok)

    def test_out_of_universe_ticker_ignored(self):
        # model returns a different, out-of-universe ticker -> distrust, default 0
        llm = MockLLM(force=lambda p: LLMResult(
            parsed={"ticker": "ZZZZ", "score": 0.9, "confidence": 0.9, "rationale": "x"},
            raw="{}"))
        ps = self._strategy(llm).evaluate(self.UNIVERSE)
        for sym in self.UNIVERSE:
            self.assertEqual(ps.signals[sym].score, 0.0)
            self.assertIn("out_of_universe_ticker", ps.signals[sym].error)

    def test_missing_news_defaults_zero_without_calling_llm(self):
        calls = {"n": 0}

        class CountingLLM(MockLLM):
            def complete_json(self, prompt):
                calls["n"] += 1
                return super().complete_json(prompt)

        strat = NewsPortfolioStrategy(llm=CountingLLM(mapping={"AAPL": (0.5, 0.5)}),
                                      universe=["AAPL"], get_info_fn=fake_get_info(n=0),
                                      audit_dir=tempfile.mkdtemp())
        ps = strat.evaluate(["AAPL"])
        self.assertEqual(ps.signals["AAPL"].score, 0.0)
        self.assertEqual(ps.signals["AAPL"].error, "no_news")
        self.assertEqual(calls["n"], 0)                 # no LLM call when no news

    def test_strategy_never_raises_on_get_info_failure(self):
        def boom(sym, as_of=None, limit=12):
            raise RuntimeError("feed down")
        strat = NewsPortfolioStrategy(llm=MockLLM(), universe=["AAPL"], get_info_fn=boom,
                                      audit_dir=tempfile.mkdtemp())
        ps = strat.evaluate(["AAPL"])                   # must not raise
        self.assertEqual(ps.signals["AAPL"].score, 0.0)


# --------------------------------------------------------------------------- #
# constructor
# --------------------------------------------------------------------------- #
def _sigs(d):
    return {t: SymbolSignal(t, score=s, confidence=c) for t, (s, c) in d.items()}


class TestConstructor(unittest.TestCase):
    def test_long_only_selects_positives_above_min_score(self):
        limits = PortfolioLimits(max_weight=0.5, max_gross=1.0, min_cash=0.0,
                                 min_positions=1, max_positions=10, min_score=0.05,
                                 weighting="equal")
        sigs = _sigs({"A": (0.6, 0.9), "B": (-0.2, 0.9), "C": (0.02, 0.9)})
        res = construct_targets(sigs, limits)
        self.assertIn("A", res.weights)
        self.assertNotIn("B", res.weights)       # negative -> excluded (long only)
        self.assertNotIn("C", res.weights)       # below min_score

    def test_max_weight_cap_pushes_excess_to_cash(self):
        limits = PortfolioLimits(max_weight=0.10, max_gross=0.95, min_cash=0.05,
                                 min_positions=1, max_positions=10, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1.0), "B": (0.8, 1.0)})
        res = construct_targets(sigs, limits)
        self.assertLessEqual(max(res.weights.values()), 0.10 + 1e-9)
        self.assertGreater(res.cash_weight, 0.5)  # most is cash because of the tight cap

    def test_max_positions_trims(self):
        limits = PortfolioLimits(max_weight=1.0, max_gross=1.0, min_cash=0.0,
                                 min_positions=1, max_positions=2, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1), "B": (0.8, 1), "C": (0.7, 1), "D": (0.6, 1)})
        res = construct_targets(sigs, limits)
        self.assertEqual(len(res.weights), 2)
        self.assertEqual(set(res.weights), {"A", "B"})   # top-2 by score

    def test_min_positions_scales_budget_down(self):
        limits = PortfolioLimits(max_weight=1.0, max_gross=1.0, min_cash=0.0,
                                 min_positions=4, max_positions=10, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1), "B": (0.8, 1)})     # only 2 of 4 wanted
        res = construct_targets(sigs, limits)
        invested = sum(res.weights.values())
        self.assertAlmostEqual(invested, 0.5, places=6)  # 2/4 of full budget
        self.assertAlmostEqual(res.cash_weight, 0.5, places=6)

    def test_gross_and_cash_respected(self):
        limits = PortfolioLimits(max_weight=0.5, max_gross=0.8, min_cash=0.2,
                                 min_positions=1, max_positions=10, weighting="confidence",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 0.9), "B": (0.7, 0.8), "C": (0.5, 0.5)})
        res = construct_targets(sigs, limits)
        self.assertLessEqual(sum(res.weights.values()), 0.8 + 1e-9)
        self.assertGreaterEqual(res.cash_weight, 0.2 - 1e-9)

    def test_exposure_multiplier_scales(self):
        limits = PortfolioLimits(max_weight=1.0, max_gross=1.0, min_cash=0.0,
                                 min_positions=1, max_positions=10, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1), "B": (0.8, 1)})
        full = construct_targets(sigs, limits, exposure_multiplier=1.0)
        half = construct_targets(sigs, limits, exposure_multiplier=0.5)
        self.assertAlmostEqual(sum(half.weights.values()),
                               0.5 * sum(full.weights.values()), places=6)

    def test_no_candidates_all_cash(self):
        limits = PortfolioLimits(min_score=0.05)
        res = construct_targets(_sigs({"A": (-0.1, 0.9), "B": (0.0, 0.9)}), limits)
        self.assertEqual(res.weights, {})
        self.assertEqual(res.cash_weight, 1.0)

    # ---- conviction gating + neutral dead-band (the exit-logic fix) ---- #
    def test_weak_held_signal_trims_to_cap_not_exit(self):
        # AAPL: score -0.10 x conf 0.50 = -0.05 conviction -> NEUTRAL (|.05| <= band .10).
        # Held at 95% -> must trim toward the 20% cap, NOT liquidate to 0%.
        limits = PortfolioLimits(max_weight=0.20, neutral_band=0.10, min_score=0.05)
        res = construct_targets(_sigs({"AAPL": (-0.10, 0.50)}), limits,
                                current_weights={"AAPL": 0.95})
        self.assertAlmostEqual(res.weights["AAPL"], 0.20, places=6)   # trimmed to cap
        self.assertNotIn("AAPL", res.selected)                        # not a fresh buy

    def test_strong_confident_negative_held_exits(self):
        # XOM-like: -0.80 x 0.90 = -0.72 conviction -> decisive exit to 0.
        limits = PortfolioLimits(neutral_band=0.10)
        res = construct_targets(_sigs({"XOM": (-0.80, 0.90)}), limits,
                                current_weights={"XOM": 0.20})
        self.assertNotIn("XOM", res.weights)
        self.assertAlmostEqual(res.cash_weight, 1.0)

    def test_gating_is_conviction_not_raw_score(self):
        # High score but low confidence -> low conviction -> NOT a candidate;
        # modest score with high confidence -> a candidate.
        limits = PortfolioLimits(neutral_band=0.10, min_score=0.05, max_weight=1.0,
                                 max_gross=1.0, min_cash=0.0, min_positions=1)
        res = construct_targets(_sigs({"A": (0.9, 0.05), "B": (0.3, 0.9)}), limits)
        self.assertEqual(res.selected, ["B"])
        self.assertNotIn("A", res.weights)

    def test_unheld_neutral_stays_flat(self):
        limits = PortfolioLimits(neutral_band=0.10)
        res = construct_targets(_sigs({"A": (0.1, 0.5)}), limits)   # conv 0.05, not held
        self.assertEqual(res.weights, {})

    def test_turnover_cap_limits_daily_move(self):
        # AAPL neutral, held 95%, would trim to 20% (move 0.75). With a 0.10 cap the
        # move is scaled so AAPL stays well above the cap this rebalance.
        limits = PortfolioLimits(max_weight=0.20, neutral_band=0.10, turnover_cap=0.10)
        res = construct_targets(_sigs({"AAPL": (-0.10, 0.50)}), limits,
                                current_weights={"AAPL": 0.95})
        # move scaled by 0.10/0.75 -> AAPL = 0.95 - 0.75*(0.10/0.75) = 0.85
        self.assertAlmostEqual(res.weights["AAPL"], 0.85, places=2)
        total_move = abs(res.weights["AAPL"] - 0.95)
        self.assertLessEqual(total_move, 0.10 + 1e-6)


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #
class TestReconciliation(unittest.TestCase):
    def test_orders_buy_sell_and_new_entry(self):
        targets = {"AAPL": 0.30, "MSFT": 0.20}
        prices = {"AAPL": 100.0, "MSFT": 50.0, "TSLA": 25.0}
        current = {"AAPL": 100, "TSLA": 40}     # hold AAPL + TSLA
        managed = {"AAPL": True, "MSFT": True, "TSLA": True}
        universe = ["AAPL", "MSFT", "TSLA"]
        orders, skipped = compute_orders(targets, 100_000, prices, current, managed, universe)
        by = {o.symbol: o for o in orders}
        # AAPL target = 0.30*100000/100 = 300 shares; have 100 -> BUY 200
        self.assertEqual(by["AAPL"].delta, 200)
        self.assertFalse(by["AAPL"].new_entry)
        # MSFT target = 0.20*100000/50 = 400; have 0 -> BUY 400, NEW entry
        self.assertEqual(by["MSFT"].delta, 400)
        self.assertTrue(by["MSFT"].new_entry)
        # TSLA target 0 (not in targets) but held+managed -> SELL all 40
        self.assertEqual(by["TSLA"].delta, -40)
        self.assertEqual(by["TSLA"].side, "SELL")

    def test_unmanaged_held_position_is_walled_off(self):
        targets = {"AAPL": 0.30}
        prices = {"AAPL": 100.0, "GME": 20.0}
        current = {"GME": 500}                  # discovered, unmanaged
        managed = {"AAPL": True, "GME": False}
        orders, skipped = compute_orders(targets, 100_000, prices, current,
                                         managed, ["AAPL"])
        syms = {o.symbol for o in orders}
        self.assertNotIn("GME", syms)           # never touched
        self.assertIn(("GME", "unmanaged_position"), skipped)

    def test_missing_price_skips(self):
        orders, skipped = compute_orders({"AAPL": 0.3}, 100_000, {"AAPL": None},
                                         {}, {"AAPL": True}, ["AAPL"])
        self.assertEqual(orders, [])
        self.assertIn(("AAPL", "no_price"), skipped)


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #
class TestBenchmarks(unittest.TestCase):
    def test_equal_weight_benchmark(self):
        start = {"A": 100.0, "B": 50.0}
        cur = {"A": 110.0, "B": 55.0}           # both +10%
        self.assertAlmostEqual(equal_weight_benchmark(100_000, start, cur), 110_000.0)
        cur2 = {"A": 120.0, "B": 50.0}          # +20% and 0% -> +10% avg
        self.assertAlmostEqual(equal_weight_benchmark(100_000, start, cur2), 110_000.0)

    def test_spy_benchmark(self):
        self.assertAlmostEqual(spy_benchmark(100_000, 400.0, 440.0), 110_000.0)
        self.assertEqual(spy_benchmark(100_000, None, 440.0), 100_000.0)  # no start -> flat

    def test_turnover(self):
        from live_portfolio import Order
        orders = [Order("A", 300, 100, 200, "BUY", 0.3),
                  Order("B", 0, 40, -40, "SELL", 0.0)]
        prices = {"A": 100.0, "B": 50.0}
        # traded notional = 200*100 + 40*50 = 22000; / 100000 = 0.22
        self.assertAlmostEqual(portfolio_turnover(orders, 100_000, prices), 0.22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
